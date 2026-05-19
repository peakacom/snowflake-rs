#![doc(
    issue_tracker_base_url = "https://github.com/mycelial/snowflake-rs/issues",
    test(no_crate_inject)
)]
#![doc = include_str!("../README.md")]
#![warn(clippy::all, clippy::pedantic)]
#![allow(
clippy::must_use_candidate,
clippy::missing_errors_doc,
clippy::module_name_repetitions,
clippy::struct_field_names,
clippy::future_not_send, // This one seems like something we should eventually fix
clippy::missing_panics_doc
)]

use std::fmt::{Display, Formatter};
use std::io;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Int32Array, Int64Array, StructArray, TimestampMicrosecondArray};
use arrow_ipc::reader::StreamReader;
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use base64::Engine;
use bytes::{Buf, Bytes};
use futures::future::try_join_all;
use regex::Regex;
use reqwest_middleware::ClientWithMiddleware;
use thiserror::Error;

// Part of public interface
pub use arrow_array::RecordBatch;
pub use arrow_schema::ArrowError;

use responses::ExecResponse;
use session::{AuthError, Session};

use crate::connection::QueryType;
use crate::connection::{Connection, ConnectionError};
use crate::requests::ExecRequest;
use crate::responses::{ExecResponseRowType, SnowflakeType};
use crate::session::AuthError::MissingEnvArgument;

pub mod connection;
#[cfg(feature = "polars")]
mod polars;
mod put;
mod requests;
mod responses;
mod session;

#[derive(Error, Debug)]
pub enum SnowflakeApiError {
    #[error(transparent)]
    RequestError(#[from] ConnectionError),

    #[error(transparent)]
    AuthError(#[from] AuthError),

    #[error(transparent)]
    ResponseDeserializationError(#[from] base64::DecodeError),

    #[error(transparent)]
    ArrowError(#[from] ArrowError),

    #[error("S3 bucket path in PUT request is invalid: `{0}`")]
    InvalidBucketPath(String),

    #[error("Couldn't extract filename from the local path: `{0}`")]
    InvalidLocalPath(String),

    #[error(transparent)]
    LocalIoError(#[from] io::Error),

    #[error(transparent)]
    ObjectStoreError(#[from] object_store::Error),

    #[error(transparent)]
    ObjectStorePathError(#[from] object_store::path::Error),

    #[error(transparent)]
    TokioTaskJoinError(#[from] tokio::task::JoinError),

    #[error("Snowflake API error. Code: `{0}`. Message: `{1}`")]
    ApiError(String, String),

    #[error("Snowflake API empty response could mean that query wasn't executed correctly or API call was faulty")]
    EmptyResponse,

    #[error("No usable rowsets were included in the response")]
    BrokenResponse,

    #[error("Following feature is not implemented yet: {0}")]
    Unimplemented(String),

    #[error("Unexpected API response")]
    UnexpectedResponse,

    #[error(transparent)]
    GlobPatternError(#[from] glob::PatternError),

    #[error(transparent)]
    GlobError(#[from] glob::GlobError),
}

/// Even if Arrow is specified as a return type non-select queries
/// will return Json array of arrays: `[[42, "answer"], [43, "non-answer"]]`.
pub struct JsonResult {
    // todo: can it _only_ be a json array of arrays or something else too?
    pub value: serde_json::Value,
    /// Field ordering matches the array ordering
    pub schema: Vec<FieldSchema>,
}

impl Display for JsonResult {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}

/// Based on the [`ExecResponseRowType`]
pub struct FieldSchema {
    pub name: String,
    // todo: is it a good idea to expose internal response struct to the user?
    pub type_: SnowflakeType,
    pub scale: Option<i64>,
    pub precision: Option<i64>,
    pub nullable: bool,
}

impl From<ExecResponseRowType> for FieldSchema {
    fn from(value: ExecResponseRowType) -> Self {
        FieldSchema {
            name: value.name,
            type_: value.type_,
            scale: value.scale,
            precision: value.precision,
            nullable: value.nullable,
        }
    }
}

/// Container for query result.
/// Arrow is returned by-default for all SELECT statements,
/// unless there is session configuration issue or it's a different statement type.
pub enum QueryResult {
    Arrow(Vec<RecordBatch>),
    Json(JsonResult),
    Empty,
}

/// Raw query result
/// Can be transformed into [`QueryResult`]
pub enum RawQueryResult {
    /// Arrow IPC chunks
    /// see: <https://arrow.apache.org/docs/format/Columnar.html#serialization-and-interprocess-communication-ipc>
    Bytes(Vec<Bytes>),
    /// Json payload is deserialized,
    /// as it's already a part of REST response
    Json(JsonResult),
    Empty,
}

impl RawQueryResult {
    pub fn deserialize_arrow(self) -> Result<QueryResult, ArrowError> {
        match self {
            RawQueryResult::Bytes(bytes) => {
                Self::flat_bytes_to_batches(bytes).map(QueryResult::Arrow)
            }
            RawQueryResult::Json(j) => Ok(QueryResult::Json(j)),
            RawQueryResult::Empty => Ok(QueryResult::Empty),
        }
    }

    fn flat_bytes_to_batches(bytes: Vec<Bytes>) -> Result<Vec<RecordBatch>, ArrowError> {
        let mut res = vec![];
        for b in bytes {
            let mut batches = Self::bytes_to_batches(b)?;
            res.append(&mut batches);
        }
        Ok(res)
    }

    fn bytes_to_batches(bytes: Bytes) -> Result<Vec<RecordBatch>, ArrowError> {
        let record_batches = StreamReader::try_new(bytes.reader(), None)?;
        record_batches
            .into_iter()
            .map(|r| r.and_then(flatten_snowflake_types))
            .collect()
    }
}

// Snowflake encodes some logical types as Arrow `Struct`s with field-level
// metadata pinning the logical type. The federation/DataFusion side expects
// the corresponding native Arrow type, so we flatten here at the decode
// boundary. Currently handles `TIMESTAMP_NTZ` (`Struct{epoch, fraction}` →
// `Timestamp(Microsecond, None)`); siblings `TIMESTAMP_LTZ`/`TIMESTAMP_TZ`
// have the same shape plus a timezone field and can be added when needed.

const SNOWFLAKE_LOGICAL_TYPE_KEY: &str = "logicalType";
const SNOWFLAKE_SCALE_KEY: &str = "scale";
const SNOWFLAKE_TIMESTAMP_NTZ: &str = "TIMESTAMP_NTZ";

fn flatten_snowflake_types(batch: RecordBatch) -> Result<RecordBatch, ArrowError> {
    let schema = batch.schema();
    let mut new_fields: Vec<Arc<Field>> = Vec::with_capacity(schema.fields().len());
    let mut new_columns: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());
    let mut changed = false;

    for (idx, field) in schema.fields().iter().enumerate() {
        let column = batch.column(idx);
        if is_snowflake_timestamp_ntz_struct(field, column) {
            let (new_field, new_column) = flatten_timestamp_ntz_column(field, column)?;
            new_fields.push(Arc::new(new_field));
            new_columns.push(new_column);
            changed = true;
        } else {
            new_fields.push(field.clone());
            new_columns.push(column.clone());
        }
    }

    if !changed {
        return Ok(batch);
    }

    let new_schema = Arc::new(Schema::new_with_metadata(
        new_fields,
        schema.metadata().clone(),
    ));
    RecordBatch::try_new(new_schema, new_columns)
}

fn is_snowflake_timestamp_ntz_struct(field: &Field, column: &ArrayRef) -> bool {
    matches!(column.data_type(), DataType::Struct(_))
        && field
            .metadata()
            .get(SNOWFLAKE_LOGICAL_TYPE_KEY)
            .is_some_and(|v| v == SNOWFLAKE_TIMESTAMP_NTZ)
}

fn flatten_timestamp_ntz_column(
    field: &Field,
    column: &ArrayRef,
) -> Result<(Field, ArrayRef), ArrowError> {
    let scale: u32 = field
        .metadata()
        .get(SNOWFLAKE_SCALE_KEY)
        .and_then(|s| s.parse().ok())
        .unwrap_or(9);

    let struct_array = column
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| {
            ArrowError::SchemaError(format!(
                "expected Struct column for TIMESTAMP_NTZ field `{}`",
                field.name()
            ))
        })?;

    let epoch = struct_array
        .column_by_name("epoch")
        .ok_or_else(|| {
            ArrowError::SchemaError(format!(
                "TIMESTAMP_NTZ struct `{}` missing `epoch` child",
                field.name()
            ))
        })?
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| {
            ArrowError::SchemaError(format!(
                "TIMESTAMP_NTZ `{}` epoch child is not Int64",
                field.name()
            ))
        })?;

    let fraction = struct_array
        .column_by_name("fraction")
        .map(|c| {
            c.as_any().downcast_ref::<Int32Array>().ok_or_else(|| {
                ArrowError::SchemaError(format!(
                    "TIMESTAMP_NTZ `{}` fraction child is not Int32",
                    field.name()
                ))
            })
        })
        .transpose()?;

    let len = struct_array.len();
    let mut builder = TimestampMicrosecondArray::builder(len);

    for i in 0..len {
        // Snowflake may signal a null TIMESTAMP_NTZ either by nulling the
        // parent struct or by nulling the `epoch` child — accept both.
        if struct_array.is_null(i) || epoch.is_null(i) {
            builder.append_null();
            continue;
        }

        let secs = epoch.value(i);
        let fr_units = fraction.map_or(0_i64, |f| {
            if f.is_null(i) {
                0_i64
            } else {
                i64::from(f.value(i))
            }
        });

        // `scale` is the precision of `fraction` in negative powers of ten
        // (scale=9 → nanoseconds, scale=6 → microseconds, etc.). Convert the
        // sub-second portion to microseconds before adding to `epoch * 1e6`.
        let frac_micros = if scale <= 6 {
            fr_units
                .checked_mul(10_i64.pow(6 - scale))
                .ok_or_else(|| overflow_err(field.name(), i))?
        } else {
            fr_units / 10_i64.pow(scale - 6)
        };

        let micros = secs
            .checked_mul(1_000_000)
            .and_then(|s| s.checked_add(frac_micros))
            .ok_or_else(|| overflow_err(field.name(), i))?;

        builder.append_value(micros);
    }

    let new_array: ArrayRef = Arc::new(builder.finish());
    let mut new_metadata = field.metadata().clone();
    new_metadata.remove(SNOWFLAKE_LOGICAL_TYPE_KEY);
    new_metadata.remove(SNOWFLAKE_SCALE_KEY);

    let new_field = Field::new(
        field.name(),
        DataType::Timestamp(TimeUnit::Microsecond, None),
        field.is_nullable(),
    )
    .with_metadata(new_metadata);

    Ok((new_field, new_array))
}

fn overflow_err(name: &str, row: usize) -> ArrowError {
    ArrowError::CastError(format!(
        "TIMESTAMP_NTZ `{name}` overflows i64 microseconds at row {row}"
    ))
}

pub struct AuthArgs {
    pub account_identifier: String,
    pub warehouse: Option<String>,
    pub database: Option<String>,
    pub schema: Option<String>,
    pub username: String,
    pub role: Option<String>,
    pub auth_type: AuthType,
}

impl AuthArgs {
    pub fn from_env() -> Result<AuthArgs, SnowflakeApiError> {
        let auth_type = if let Ok(password) = std::env::var("SNOWFLAKE_PASSWORD") {
            Ok(AuthType::Password(PasswordArgs { password }))
        } else if let Ok(private_key_pem) = std::env::var("SNOWFLAKE_PRIVATE_KEY") {
            Ok(AuthType::Certificate(CertificateArgs { private_key_pem }))
        } else {
            Err(MissingEnvArgument(
                "SNOWFLAKE_PASSWORD or SNOWFLAKE_PRIVATE_KEY".to_owned(),
            ))
        };

        Ok(AuthArgs {
            account_identifier: std::env::var("SNOWFLAKE_ACCOUNT")
                .map_err(|_| MissingEnvArgument("SNOWFLAKE_ACCOUNT".to_owned()))?,
            warehouse: std::env::var("SNOWLFLAKE_WAREHOUSE").ok(),
            database: std::env::var("SNOWFLAKE_DATABASE").ok(),
            schema: std::env::var("SNOWFLAKE_SCHEMA").ok(),
            username: std::env::var("SNOWFLAKE_USER")
                .map_err(|_| MissingEnvArgument("SNOWFLAKE_USER".to_owned()))?,
            role: std::env::var("SNOWFLAKE_ROLE").ok(),
            auth_type: auth_type?,
        })
    }
}

pub enum AuthType {
    Password(PasswordArgs),
    Certificate(CertificateArgs),
}

pub struct PasswordArgs {
    pub password: String,
}

pub struct CertificateArgs {
    pub private_key_pem: String,
}

#[must_use]
pub struct SnowflakeApiBuilder {
    pub auth: AuthArgs,
    client: Option<ClientWithMiddleware>,
}

impl SnowflakeApiBuilder {
    pub fn new(auth: AuthArgs) -> Self {
        Self { auth, client: None }
    }

    pub fn with_client(mut self, client: ClientWithMiddleware) -> Self {
        self.client = Some(client);
        self
    }

    pub fn build(self) -> Result<SnowflakeApi, SnowflakeApiError> {
        let connection = match self.client {
            Some(client) => Arc::new(Connection::new_with_middware(client)),
            None => Arc::new(Connection::new()?),
        };

        let session = match self.auth.auth_type {
            AuthType::Password(args) => Session::password_auth(
                Arc::clone(&connection),
                &self.auth.account_identifier,
                self.auth.warehouse.as_deref(),
                self.auth.database.as_deref(),
                self.auth.schema.as_deref(),
                &self.auth.username,
                self.auth.role.as_deref(),
                &args.password,
            ),
            AuthType::Certificate(args) => Session::cert_auth(
                Arc::clone(&connection),
                &self.auth.account_identifier,
                self.auth.warehouse.as_deref(),
                self.auth.database.as_deref(),
                self.auth.schema.as_deref(),
                &self.auth.username,
                self.auth.role.as_deref(),
                &args.private_key_pem,
            ),
        };

        let account_identifier = self.auth.account_identifier.to_uppercase();

        Ok(SnowflakeApi::new(
            Arc::clone(&connection),
            session,
            account_identifier,
        ))
    }
}

/// Snowflake API, keeps connection pool and manages session for you
pub struct SnowflakeApi {
    connection: Arc<Connection>,
    session: Session,
    account_identifier: String,
}

impl SnowflakeApi {
    /// Create a new `SnowflakeApi` object with an existing connection and session.
    pub fn new(connection: Arc<Connection>, session: Session, account_identifier: String) -> Self {
        Self {
            connection,
            session,
            account_identifier,
        }
    }
    /// Initialize object with password auth. Authentication happens on the first request.
    pub fn with_password_auth(
        account_identifier: &str,
        warehouse: Option<&str>,
        database: Option<&str>,
        schema: Option<&str>,
        username: &str,
        role: Option<&str>,
        password: &str,
    ) -> Result<Self, SnowflakeApiError> {
        let connection = Arc::new(Connection::new()?);

        let session = Session::password_auth(
            Arc::clone(&connection),
            account_identifier,
            warehouse,
            database,
            schema,
            username,
            role,
            password,
        );

        let account_identifier = account_identifier.to_uppercase();
        Ok(Self::new(
            Arc::clone(&connection),
            session,
            account_identifier,
        ))
    }

    /// Initialize object with private certificate auth. Authentication happens on the first request.
    pub fn with_certificate_auth(
        account_identifier: &str,
        warehouse: Option<&str>,
        database: Option<&str>,
        schema: Option<&str>,
        username: &str,
        role: Option<&str>,
        private_key_pem: &str,
    ) -> Result<Self, SnowflakeApiError> {
        let connection = Arc::new(Connection::new()?);

        let session = Session::cert_auth(
            Arc::clone(&connection),
            account_identifier,
            warehouse,
            database,
            schema,
            username,
            role,
            private_key_pem,
        );

        let account_identifier = account_identifier.to_uppercase();
        Ok(Self::new(
            Arc::clone(&connection),
            session,
            account_identifier,
        ))
    }

    pub fn from_env() -> Result<Self, SnowflakeApiError> {
        SnowflakeApiBuilder::new(AuthArgs::from_env()?).build()
    }

    /// Closes the current session, this is necessary to clean up temporary objects (tables, functions, etc)
    /// which are Snowflake session dependent.
    /// If another request is made the new session will be initiated.
    pub async fn close_session(&mut self) -> Result<(), SnowflakeApiError> {
        self.session.close().await?;
        Ok(())
    }

    /// Execute a single query against API.
    /// If statement is PUT, then file will be uploaded to the Snowflake-managed storage
    pub async fn exec(&self, sql: &str) -> Result<QueryResult, SnowflakeApiError> {
        let raw = self.exec_raw(sql).await?;
        let res = raw.deserialize_arrow()?;
        Ok(res)
    }

    /// Executes a single query against API.
    /// If statement is PUT, then file will be uploaded to the Snowflake-managed storage
    /// Returns raw bytes in the Arrow response
    pub async fn exec_raw(&self, sql: &str) -> Result<RawQueryResult, SnowflakeApiError> {
        let put_re = Regex::new(r"(?i)^(?:/\*.*\*/\s*)*put\s+").unwrap();

        // put commands go through a different flow and result is side-effect
        if put_re.is_match(sql) {
            log::info!("Detected PUT query");
            self.exec_put(sql).await.map(|()| RawQueryResult::Empty)
        } else {
            self.exec_arrow_raw(sql).await
        }
    }

    async fn exec_put(&self, sql: &str) -> Result<(), SnowflakeApiError> {
        let resp = self
            .run_sql::<ExecResponse>(sql, QueryType::JsonQuery)
            .await?;
        log::debug!("Got PUT response: {resp:?}");

        match resp {
            ExecResponse::Query(_) => Err(SnowflakeApiError::UnexpectedResponse),
            ExecResponse::PutGet(pg) => put::put(pg).await,
            ExecResponse::Error(e) => Err(SnowflakeApiError::ApiError(
                e.data.error_code,
                e.message.unwrap_or_default(),
            )),
        }
    }

    /// Useful for debugging to get the straight query response
    #[cfg(debug_assertions)]
    pub async fn exec_response(&mut self, sql: &str) -> Result<ExecResponse, SnowflakeApiError> {
        self.run_sql::<ExecResponse>(sql, QueryType::ArrowQuery)
            .await
    }

    /// Useful for debugging to get raw JSON response
    #[cfg(debug_assertions)]
    pub async fn exec_json(&mut self, sql: &str) -> Result<serde_json::Value, SnowflakeApiError> {
        self.run_sql::<serde_json::Value>(sql, QueryType::JsonQuery)
            .await
    }

    async fn exec_arrow_raw(&self, sql: &str) -> Result<RawQueryResult, SnowflakeApiError> {
        let resp = self
            .run_sql::<ExecResponse>(sql, QueryType::ArrowQuery)
            .await?;
        log::debug!("Got query response: {resp:?}");

        let resp = match resp {
            // processable response
            ExecResponse::Query(qr) => Ok(qr),
            ExecResponse::PutGet(_) => Err(SnowflakeApiError::UnexpectedResponse),
            ExecResponse::Error(e) => Err(SnowflakeApiError::ApiError(
                e.data.error_code,
                e.message.unwrap_or_default(),
            )),
        }?;

        // if response was empty, base64 data is empty string
        // todo: still return empty arrow batch with proper schema? (schema always included)
        if resp.data.returned == 0 {
            log::debug!("Got response with 0 rows");
            Ok(RawQueryResult::Empty)
        } else if let Some(value) = resp.data.rowset {
            log::debug!("Got JSON response");
            // NOTE: json response could be chunked too. however, go clients should receive arrow by-default,
            // unless user sets session variable to return json. This case was added for debugging and status
            // information being passed through that fields.
            Ok(RawQueryResult::Json(JsonResult {
                value,
                schema: resp.data.rowtype.into_iter().map(Into::into).collect(),
            }))
        } else if let Some(base64) = resp.data.rowset_base64 {
            // fixme: is it possible to give streaming interface?
            let mut chunks = try_join_all(resp.data.chunks.iter().map(|chunk| {
                self.connection
                    .get_chunk(&chunk.url, &resp.data.chunk_headers)
            }))
            .await?;

            // fixme: should base64 chunk go first?
            // fixme: if response is chunked is it both base64 + chunks or just chunks?
            if !base64.is_empty() {
                log::debug!("Got base64 encoded response");
                let bytes = Bytes::from(base64::engine::general_purpose::STANDARD.decode(base64)?);
                chunks.push(bytes);
            }

            Ok(RawQueryResult::Bytes(chunks))
        } else {
            Err(SnowflakeApiError::BrokenResponse)
        }
    }

    async fn run_sql<R: serde::de::DeserializeOwned>(
        &self,
        sql_text: &str,
        query_type: QueryType,
    ) -> Result<R, SnowflakeApiError> {
        log::debug!("Executing: {sql_text}");

        let parts = self.session.get_token().await?;

        let body = ExecRequest {
            sql_text: sql_text.to_string(),
            async_exec: false,
            sequence_id: parts.sequence_id,
            is_internal: false,
        };

        let resp = self
            .connection
            .request::<R>(
                query_type,
                &self.account_identifier,
                &[],
                Some(&parts.session_token_auth_header),
                body,
            )
            .await?;

        Ok(resp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use arrow_schema::{Field, Fields};

    fn ntz_struct_column(
        epochs: Vec<Option<i64>>,
        fractions: Vec<Option<i32>>,
    ) -> (Field, ArrayRef) {
        let epoch = Arc::new(Int64Array::from(epochs)) as ArrayRef;
        let fraction = Arc::new(Int32Array::from(fractions)) as ArrayRef;
        let fields = Fields::from(vec![
            Field::new("epoch", DataType::Int64, true),
            Field::new("fraction", DataType::Int32, true),
        ]);
        let struct_array =
            StructArray::new(fields.clone(), vec![epoch, fraction], None);

        let mut meta = HashMap::new();
        meta.insert(SNOWFLAKE_LOGICAL_TYPE_KEY.to_string(), SNOWFLAKE_TIMESTAMP_NTZ.to_string());
        meta.insert(SNOWFLAKE_SCALE_KEY.to_string(), "9".to_string());

        let field = Field::new("ts", DataType::Struct(fields), true).with_metadata(meta);
        (field, Arc::new(struct_array))
    }

    #[test]
    fn flatten_timestamp_ntz_scale_9() {
        // 2026-01-02 03:04:05.123456789 UTC = 1767322445 seconds + 123_456_789 ns
        // Expected micros: 1767322445_000000 + 123_456 = 1767322445_123456
        let (field, column) = ntz_struct_column(
            vec![Some(1_767_322_445), None, Some(0)],
            vec![Some(123_456_789), None, Some(0)],
        );
        let schema = Arc::new(Schema::new(vec![field]));
        let batch = RecordBatch::try_new(schema, vec![column]).unwrap();

        let flat = flatten_snowflake_types(batch).unwrap();
        assert_eq!(
            flat.schema().field(0).data_type(),
            &DataType::Timestamp(TimeUnit::Microsecond, None)
        );
        let ts = flat
            .column(0)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .expect("flattened to TimestampMicrosecondArray");
        assert_eq!(ts.value(0), 1_767_322_445_123_456);
        assert!(ts.is_null(1));
        assert_eq!(ts.value(2), 0);
        // Metadata pinning should be consumed by the flatten step.
        assert!(!flat
            .schema()
            .field(0)
            .metadata()
            .contains_key(SNOWFLAKE_LOGICAL_TYPE_KEY));
    }

    #[test]
    fn flatten_timestamp_ntz_parent_null_buffer() {
        // Same shape, but the null is signalled on the parent struct rather
        // than the `epoch` child — Snowflake uses either depending on the
        // batch source, so both must yield `is_null` rows.
        use arrow_buffer::NullBuffer;
        let epoch = Arc::new(Int64Array::from(vec![1_767_322_445_i64, 0, 0])) as ArrayRef;
        let fraction = Arc::new(Int32Array::from(vec![123_456_789_i32, 0, 0])) as ArrayRef;
        let fields = Fields::from(vec![
            Field::new("epoch", DataType::Int64, true),
            Field::new("fraction", DataType::Int32, true),
        ]);
        let nulls = NullBuffer::from(vec![true, false, true]);
        let struct_array = StructArray::new(fields.clone(), vec![epoch, fraction], Some(nulls));

        let mut meta = HashMap::new();
        meta.insert(
            SNOWFLAKE_LOGICAL_TYPE_KEY.to_string(),
            SNOWFLAKE_TIMESTAMP_NTZ.to_string(),
        );
        meta.insert(SNOWFLAKE_SCALE_KEY.to_string(), "9".to_string());
        let field = Field::new("ts", DataType::Struct(fields), true).with_metadata(meta);
        let schema = Arc::new(Schema::new(vec![field]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(struct_array)]).unwrap();

        let flat = flatten_snowflake_types(batch).unwrap();
        let ts = flat
            .column(0)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap();
        assert_eq!(ts.value(0), 1_767_322_445_123_456);
        assert!(ts.is_null(1));
        assert_eq!(ts.value(2), 0);
    }

    #[test]
    fn flatten_pass_through_when_no_marker() {
        let epoch = Arc::new(Int64Array::from(vec![1_i64, 2, 3])) as ArrayRef;
        let field = Field::new("plain", DataType::Int64, false);
        let schema = Arc::new(Schema::new(vec![field]));
        let batch = RecordBatch::try_new(schema.clone(), vec![epoch.clone()]).unwrap();

        let flat = flatten_snowflake_types(batch).unwrap();
        assert_eq!(flat.schema(), schema);
        assert_eq!(flat.column(0).as_ref(), epoch.as_ref());
    }
}

use super::{
    AccountedEngineData, FailureKind, OperationFailure, Resource, ResourceExhausted, TaskLimits,
};
use crate::engine_data::{GetData, RowVisitor};
use crate::schema::{ColumnName, DataType};

pub(super) fn preflight_pm(
    batch: &dyn AccountedEngineData,
    limits: &TaskLimits,
    mut admit: impl FnMut(usize) -> Result<(), OperationFailure>,
) -> Result<usize, OperationFailure> {
    if batch.len() > 1 {
        return Err(OperationFailure::malformed_response());
    }
    let fixed = super::json_visitor_allocation::pm_fixed_peak()
        .ok_or_else(|| exhausted(Resource::MetadataAllocatedBytes, limits))?;
    check(Resource::MetadataAllocatedBytes, fixed, limits)?;
    check(Resource::TaskStateBytes, fixed, limits)?;
    admit(fixed)?;
    // These borrowed views reject feature/partition owners before the ordinary
    // action visitors call ListItem::materialize. Preserve their action semantics.
    let mut protocol = ProtocolPreflight { failure: None };
    protocol.visit_rows_of(batch).map_err(engine)?;
    if let Some(failure) = protocol.failure {
        return Err(failure);
    }
    let mut visitor = SchemaPreflight {
        limits,
        failure: None,
        owned_bytes: 0,
    };
    visitor.visit_rows_of(batch).map_err(engine)?;
    if let Some(failure) = visitor.failure {
        return Err(failure);
    }
    let bytes = fixed
        .checked_add(visitor.owned_bytes)
        .ok_or_else(|| exhausted(Resource::MetadataAllocatedBytes, limits))?;
    check(Resource::MetadataAllocatedBytes, bytes, limits)?;
    check(Resource::TaskStateBytes, bytes, limits)?;
    admit(bytes)?;
    Ok(bytes)
}

struct ProtocolPreflight {
    failure: Option<OperationFailure>,
}
impl RowVisitor for ProtocolPreflight {
    fn selected_column_names_and_types(&self) -> (&'static [ColumnName], &'static [DataType]) {
        crate::actions::visitors::PROTOCOL_LEAVES.as_ref()
    }
    fn visit<'a>(
        &mut self,
        rows: usize,
        getters: &[&'a dyn GetData<'a>],
    ) -> crate::DeltaResult<()> {
        if rows > 1 || getters.len() != 4 {
            return Err(crate::Error::generic("malformed protocol page"));
        }
        if rows == 0 {
            return Ok(());
        }
        let reader = getters[0].get_int(0, "protocol.minReaderVersion")?;
        if reader.is_none() {
            return Ok(());
        }
        let writer = getters[1].get_int(0, "protocol.minWriterVersion")?;
        if reader != Some(1)
            || writer.is_some_and(|version| version > 2)
            || getters[2]
                .get_list(0, "protocol.readerFeatures")?
                .is_some_and(|list| !list.is_empty())
            || getters[3]
                .get_list(0, "protocol.writerFeatures")?
                .is_some_and(|list| !list.is_empty())
        {
            self.failure = Some(engine(crate::Error::unsupported(
                "table protocol/features are outside JSON operation tasks",
            )));
        }
        Ok(())
    }
}

struct SchemaPreflight<'a> {
    limits: &'a TaskLimits,
    failure: Option<OperationFailure>,
    owned_bytes: usize,
}
impl RowVisitor for SchemaPreflight<'_> {
    fn selected_column_names_and_types(&self) -> (&'static [ColumnName], &'static [DataType]) {
        crate::actions::visitors::METADATA_LEAVES.as_ref()
    }
    fn visit<'a>(
        &mut self,
        rows: usize,
        getters: &[&'a dyn GetData<'a>],
    ) -> crate::DeltaResult<()> {
        if rows > 1 || getters.len() != 9 {
            return Err(crate::Error::generic("malformed metadata page"));
        }
        if rows == 0 || getters[0].get_str(0, "metaData.id")?.is_none() {
            return Ok(());
        }
        if getters[6]
            .get_list(0, "metaData.partitionColumns")?
            .is_some_and(|list| !list.is_empty())
        {
            self.failure = Some(engine(crate::Error::unsupported(
                "partitioned tables are outside JSON operation tasks",
            )));
        } else if getters[3]
            .get_str(0, "metaData.format.provider")?
            .is_some_and(|provider| provider != "parquet")
        {
            self.failure = Some(engine(crate::Error::unsupported(
                "JSON operation tasks require Parquet data files",
            )));
        } else if let Some(schema) = getters[5].get_str(0, "metaData.schemaString")? {
            self.failure = preflight_schema_json(schema, self.limits).err();
        }
        if self.failure.is_none() {
            let mut owned = Some(0usize);
            for index in [0, 1, 2, 3, 5] {
                if let Some(value) = getters[index].get_str(0, "metaData string")? {
                    owned = owned.and_then(|n| n.checked_add(value.len()));
                }
            }
            if let Some(map) = getters[8].get_map(0, "metaData.configuration")? {
                // materialize reserves all source slots, even null-valued or
                // duplicate entries; count every cloned key/value before dedup.
                owned = owned.and_then(|n| {
                    n.checked_add(super::json_schema_shape::hash_peak::<(String, String)>(
                        map.entry_count(),
                    )?)
                });
                for (key, value) in map.entries() {
                    if let Some(value) = value {
                        owned =
                            owned.and_then(|n| n.checked_add(key.len())?.checked_add(value.len()));
                    }
                }
            }
            match owned {
                Some(bytes) => self.owned_bytes = bytes,
                None => {
                    self.failure = Some(exhausted(Resource::MetadataAllocatedBytes, self.limits))
                }
            }
        }
        Ok(())
    }
}

// Allocation-free lexical preflight, before the existing serde schema decoder. The decoder
// remains responsible for full syntax and schema semantics. Counting punctuation/string starts
// conservatively bounds decoded nodes even for unknown metadata; escaped quotes are not tokens.
pub(super) fn preflight_schema_json(
    json: &str,
    limits: &TaskLimits,
) -> Result<(), OperationFailure> {
    check(Resource::PartialJsonBytes, json.len(), limits)?;
    check(Resource::WorkUnits, json.len(), limits)?;
    let mut depth = 0usize;
    let mut nodes = 0usize;
    let mut quoted = false;
    let mut escaped = false;
    for byte in json.bytes() {
        if quoted {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                quoted = false;
            }
            continue;
        }
        match byte {
            b'"' => {
                quoted = true;
                nodes += 1;
            }
            b'{' | b'[' => {
                depth += 1;
                nodes += 1;
            }
            b'}' | b']' => {
                depth = depth
                    .checked_sub(1)
                    .ok_or_else(OperationFailure::malformed_response)?;
            }
            b':' | b',' => {
                nodes += 1;
            }
            _ => {}
        }
        check(Resource::SchemaDepth, depth, limits)?;
        check(Resource::SchemaNodes, nodes, limits)?;
    }
    if quoted || depth != 0 {
        return Err(OperationFailure::malformed_response());
    }
    Ok(())
}

pub(super) fn check(
    resource: Resource,
    observed: usize,
    limits: &TaskLimits,
) -> Result<(), OperationFailure> {
    if observed > limits.limit(resource) {
        return Err(ResourceExhausted {
            resource,
            limit: limits.limit(resource),
            observed,
        }
        .into());
    }
    Ok(())
}
pub(super) fn exhausted(resource: Resource, limits: &TaskLimits) -> OperationFailure {
    ResourceExhausted {
        resource,
        limit: limits.limit(resource),
        observed: usize::MAX,
    }
    .into()
}
pub(super) fn engine(error: crate::Error) -> OperationFailure {
    OperationFailure::new(FailureKind::Engine, error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine_data::{ListItem, StringArrayAccessor};
    struct UnreadableList;
    impl StringArrayAccessor for UnreadableList {
        fn len(&self) -> usize {
            usize::MAX
        }
        fn value(&self, _: usize) -> &str {
            panic!("rejected list element must not be materialized")
        }
        fn is_valid(&self, _: usize) -> bool {
            true
        }
    }
    impl<'a> GetData<'a> for UnreadableList {
        fn get_list(&'a self, _: usize, _: &str) -> crate::DeltaResult<Option<ListItem<'a>>> {
            Ok(Some(ListItem::new(self, 0..usize::MAX)))
        }
    }
    struct Int(i32);
    impl<'a> GetData<'a> for Int {
        fn get_int(&'a self, _: usize, _: &str) -> crate::DeltaResult<Option<i32>> {
            Ok(Some(self.0))
        }
    }
    struct Text(&'static str);
    impl<'a> GetData<'a> for Text {
        fn get_str(&'a self, _: usize, _: &str) -> crate::DeltaResult<Option<&'a str>> {
            Ok(Some(self.0))
        }
    }
    #[test]
    fn protocol_features_reject_before_accessing_or_cloning_elements() {
        let getters: [&dyn GetData<'_>; 4] = [&Int(1), &Int(2), &UnreadableList, &()];
        let mut preflight = ProtocolPreflight { failure: None };
        preflight.visit(1, &getters).unwrap();
        assert!(matches!(
            preflight.failure.unwrap().into_error(),
            crate::Error::Unsupported(_)
        ));
    }
    #[test]
    fn metadata_partitions_reject_before_accessing_or_cloning_elements() {
        let getters: [&dyn GetData<'_>; 9] = [
            &Text("id"),
            &(),
            &(),
            &(),
            &(),
            &(),
            &UnreadableList,
            &(),
            &(),
        ];
        let limits = TaskLimits::qualification();
        let mut preflight = SchemaPreflight {
            limits: &limits,
            failure: None,
            owned_bytes: 0,
        };
        preflight.visit(1, &getters).unwrap();
        assert!(matches!(
            preflight.failure.unwrap().into_error(),
            crate::Error::Unsupported(_)
        ));
    }
}

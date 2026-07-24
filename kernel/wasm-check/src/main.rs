// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::sync::Arc;

use buoyant_kernel::arrow::array::RecordBatch;
use buoyant_kernel::arrow::datatypes::Schema;
use buoyant_kernel::engine::arrow_data::ArrowEngineData;
use buoyant_kernel::engine::sync::SyncEngine;
use buoyant_kernel::object_store::memory::InMemory;
use buoyant_kernel::object_store::DynObjectStore;
use buoyant_kernel::Engine;

fn main() {
    let store: Arc<DynObjectStore> = Arc::new(InMemory::new());
    let engine = SyncEngine::new_with_store(store);
    let data = ArrowEngineData::new(RecordBatch::new_empty(Arc::new(Schema::empty())));

    let _ = engine.evaluation_handler();
    assert_eq!(data.record_batch().num_rows(), 0);
}

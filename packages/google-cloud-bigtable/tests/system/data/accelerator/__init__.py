# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pre-release integration tests for the Bigtable accelerator.

Every test in this package exercises the *real* shipped path — the real
``BigtableDataClient`` with ``use_accelerator`` set, the real
``AcceleratorDaemon`` spawning the real bundled Go binary, the real UDS
``_AcceleratorClient``, and (for correctness/stress/backend-error tests) a real
Bigtable instance. Error conditions are induced through real inputs (a bogus
daemon binary, killing the real daemon process, hitting real backend error
conditions), never by substituting a production component with a fake.

The one place a non-production binary appears is the controlled-binary factory in
``_harness`` (bad / slow daemons). Those are fed as *inputs* to the real
``AcceleratorDaemon`` to deterministically drive its start-failure and
startup-race code paths, which a healthy binary cannot exercise.
"""

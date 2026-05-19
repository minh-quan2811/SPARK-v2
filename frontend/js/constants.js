/**
 * constants.js
 * Shared lookup tables used across dashboard modules.
 */

'use strict';

const AGENT_META = {
  cv_agent:         { label: 'CV Agent',        icon: 'ti-file-cv'   },
  job_agent:        { label: 'Job Agent',        icon: 'ti-briefcase' },
  curriculum_agent: { label: 'Curriculum Agent', icon: 'ti-school'    },
  plan_agent:       { label: 'Plan Agent',       icon: 'ti-map'       },
};

/** Maps SSE node names → thought bubble type */
const NODE_TYPE = {
  tool_call:   'tool',
  tool_result: 'result',
  query_error: 'err',
  agent_error: 'err',
  json_error:  'err',
};

/** Maps SSE node names → human-readable label */
const NODE_LABEL = {
  extract_text:    'Reading PDF',
  llm_extraction:  'AI extraction',
  parse_json:      'Parsing response',
  validate_output: 'Validating data',
  prepare_query:   'Building query',
  run_agent:       'Agent activity',
  format_results:  'Formatting',
  generate_cypher: 'Generating Cypher',
  execute_query:   'Running query',
  query_error:     'Query error',
  agent_error:     'Agent error',
  generate_plan:   'Generating roadmap',
  tool_call:       'Tool call',
  tool_result:     'Tool result',
};

/** Maps thought type → Tabler icon class */
const THOUGHT_ICON = {
  step:   'ti-arrow-right',
  tool:   'ti-plug-connected',
  result: 'ti-circle-check',
  err:    'ti-alert-triangle',
};
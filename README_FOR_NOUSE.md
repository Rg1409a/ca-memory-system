# Sanitized Version for Nouse Submission

This is a sanitized copy of nouse_hermes safe for external sharing.

## Changes from original:

1. **Test data replaced**: 
   - `test_data/synthetic_test_data.json` (generic entity_X placeholders)
   - `test_data/synthetic_benchmark_results.json` (anonymized metrics)

2. **Example paths updated**: All references point to sanitized test data

3. **No domain-specific content**: Removed MEMS/stiction terminology from examples

## Original project: /home/adc/nouse_hermes
## This sanitized copy: /home/adc/pm_expert_system_sanitized

## To submit to Nouse:
```bash
cd /home/adc/pm_expert_system_sanitized
# Verify tests pass
python -m pytest tests/
# Then share this directory
```

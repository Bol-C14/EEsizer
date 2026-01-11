# Troubleshooting

## ngspice not found
- Install ngspice or set `NGSPICE_PATH`
- In container: ensure `apt-get install ngspice` is present

## Simulation outputs missing
- Check simulator logs in the run folder
- Confirm deck builder produced the expected outputs
- Confirm output file names match what metric calculators expect

## Metrics look wrong / inconsistent
- Verify registry definitions (units, log base, column selection)
- Add/extend unit tests for the metric
- See legacy notes: `legacy/docs/metrics.md` and `legacy/docs/code_review_2025-12-31.md`

## Patch rejected too often
- Check ParamSpace bounds and frozen flags
- Reduce allowed step sizes
- Improve policy prompt: show constraints clearly

## LLM output not parseable
- Enforce JSON schema validation
- Reduce prompt ambiguity; explicitly say “Output JSON only”

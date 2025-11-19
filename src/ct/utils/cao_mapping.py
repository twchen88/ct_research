def build_cao_mapping_from_weekly_encoding(config: Dict[str, Any], meta: Dict[str, Any]) -> Tuple[Dict[str, List[str]], int, List[str]]:
    W = config["time"]["lookback_weeks"]
    domains: List[int] = list(meta["domains"])
    K = len(domains)
    covs: List[str] = list(meta.get("x_week_feature_names", []))

    # Context (scores) with weekly lags 1..W
    context_scores = [f"score_domain_{d}_lag_{k}" for k in range(1, W + 1) for d in domains]
    # Optional extras (pair + covariates), if you want to name them for inspection
    context_pair_presence = [f"present_domain_{d}_lag_{k}" for k in range(1, W + 1) for d in domains]
    context_pair_inv1m    = [f"inv1m_domain_{d}_lag_{k}"  for k in range(1, W + 1) for d in domains]
    context_covariates    = [f"{feat}_lag_{k}" for k in range(1, W + 1) for feat in covs]

    # Final context (you can include only scores if you prefer)
    context = list(context_scores) + context_pair_presence + context_pair_inv1m + context_covariates

    # Actions: one per domain (binary selection), lag_0 convention
    action_names = [f"action_domain_{d}_lag_0" for d in domains]
    Da = len(action_names)  # == K
    print("Da in build_cao_mapping_from_weekly_encoding:", Da)

    # Outcome: global or per-domain
    if config["targets"]["outcome"] == "global":
        outcome = ["global_score_next_week"]
    else:
        outcome = [f"score_domain_{d}_next_week" for d in domains]

    cao_mapping = {"context": context, "actions": action_names, "outcome": outcome}
    return cao_mapping, Da, action_names
def unroll_predictor_weekly(
    config: Dict[str, Any],
    model,                               # trained Predictor (PyTorch)
    prescriptor: ZeroPrescriptor,
    E_pairs: np.ndarray,                 # [U, T, K, 2]
    X_week: np.ndarray,                  # [U, T, D_x]
    meta: Dict[str, Any],
    user_idx: int,
    statics_row: Optional[np.ndarray] = None,  # [Ds] or None
    device: Optional[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Weekly unroll for `lookahead_weeks`: predict next week, treat it as observed, append, and repeat.
    Returns:
      pred_df: DataFrame with week_start and per-domain predicted scores + global_score
      pred_scores: (lookahead_weeks, K)
      pred_actions: (lookahead_weeks, Da) (zeros)
    """
    W = config["time"]["lookback_weeks"]
    H = config["time"]["lookahead_weeks"]
    week_freq = meta["week_freq"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model.eval(); model.to(device)

    U, T, K, _ = E_pairs.shape
    D_x = X_week.shape[2]
    Dx_total = 2 * K + D_x

    E_u = E_pairs[user_idx].copy()     # [T, K, 2]
    X_u = X_week[user_idx].copy()      # [T, D_x]
    weeks_u: pd.DatetimeIndex = meta["weeks_per_user"][user_idx]
    if len(weeks_u) < W:
        raise ValueError(f"User {user_idx} has only {len(weeks_u)} weeks; need at least lookback={W}.")

    Da = len(getattr(prescriptor, "action_names", []))
    pred_scores = np.zeros((H, K), dtype=np.float32)
    pred_actions = np.zeros((H, Da), dtype=np.float32)

    domain_cols = [f"domain_{d}" for d in meta["domains"]]
    out_rows: List[Dict[str, Any]] = []

    # last week_start
    last_week = pd.Timestamp(weeks_u[-1])
    last_x_cov = X_u[-1, :].copy()

    if statics_row is not None:
        s_static = torch.from_numpy(statics_row.reshape(1, -1)).float().to(device)
    else:
        s_static = torch.zeros((1, 0), dtype=torch.float32, device=device)

    for step in range(H):
        t_end = E_u.shape[0]
        t0, t1 = t_end - W, t_end
        # Build window tensors
        pres_win = E_u[t0:t1, :, 0]
        inv_win  = E_u[t0:t1, :, 1]
        y_hist_np = pres_win * (1.0 - inv_win)                   # [W,K]
        E_flat_win = E_u[t0:t1].reshape(W, 2 * K)                # [W,2K]
        x_hist_np = np.concatenate([E_flat_win, X_u[t0:t1, :]], axis=-1)  # [W, 2K + D_x]

        y_hist = torch.from_numpy(y_hist_np).unsqueeze(0).float().to(device)
        x_hist = torch.from_numpy(x_hist_np).unsqueeze(0).float().to(device)

        # Zero actions for next week
        context_df = pd.DataFrame({"week_start": [last_week + pd.offsets.Week(1, weekday=0)]})
        presc_df = prescriptor.prescribe(context_df)  # zeros
        if Da > 0:
            a_now_np = presc_df.iloc[0].to_numpy(dtype=np.float32).reshape(1, Da)
            pred_actions[step, :] = a_now_np[0]
            a_now = torch.from_numpy(a_now_np).float().to(device)
        else:
            a_now = torch.zeros((1, 0), dtype=torch.float32, device=device)

        user_idx_t = torch.tensor([user_idx], dtype=torch.long, device=device)

        with torch.no_grad():
            preds = model(y_hist, x_hist, a_now, s_static, user_idx_t)
            y_hat = np.clip(preds["mu"].squeeze(0).cpu().numpy(), 0.0, 1.0)   # [K]
            pred_scores[step, :] = y_hat

        # Append next week as observed
        next_week = last_week + pd.Timedelta(days=7)
        next_pairs = np.stack([np.ones(K, dtype=np.float32), 1.0 - y_hat.astype(np.float32)], axis=-1)  # [K,2]
        E_u = np.concatenate([E_u, next_pairs.reshape(1, K, 2)], axis=0)

        # Advance covariates
        next_x_cov = advance_weekly_covariates(last_x_cov)
        X_u = np.concatenate([X_u, next_x_cov.reshape(1, D_x)], axis=0)

        # Output row
        row = {"week_start": next_week}
        row.update({c: v for c, v in zip(domain_cols, y_hat.tolist())})
        row["global_score"] = float(np.mean(y_hat))
        out_rows.append(row)

        last_week = next_week
        last_x_cov = next_x_cov

    pred_df = pd.DataFrame(out_rows, columns=["week_start"] + domain_cols + ["global_score"])
    return pred_df, pred_scores, pred_actions
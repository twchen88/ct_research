
def visualize_observed_domain_counts():
    if len(ds_train) > 0:
        m_counts = [int(ds_train.M[u, t].sum()) for (u, t) in ds_train.indices]

        plt.figure()
        plt.hist(m_counts, bins=np.arange(-0.5, ds_train.K + 1.5, 1))
        plt.title("Observed domains per anchor (train, weekly)")
        plt.xlabel("# observed domains at target week")
        plt.ylabel("Count of anchors")
        plt.tight_layout()
        plt.show()
    else:
        print("No training anchors available to visualize target observation counts.")

def visualize_unrolled_global_score():
    plt.figure(figsize=(8,4))
    plt.plot(pred_df["week_start"], pred_df["global_score"], marker="o")
    plt.title(f"User {meta['users'][user_idx]} — unrolled global score (zero actions)")
    plt.xlabel("Date"); plt.ylabel("Global score (mean of domains)")
    plt.xticks(rotation=45); plt.tight_layout(); plt.show()

def visualize_user_weekly_trajectory(
    config: Dict[str, Any],
    model,
    E_pairs: np.ndarray,           # [U, T_max, K, 2]
    X_week: np.ndarray,            # [U, T_max, D_x]
    Y: np.ndarray,                 # [U, T_max, K]
    M_target: np.ndarray,          # [U, T_max, K]
    meta: Dict[str, Any],
    prescriptor,
    statics: Optional[np.ndarray] = None,  # [U, Ds] or None
    user_idx: Optional[int] = None,
    device: Optional[str] = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    W = config["time"]["lookback_weeks"]
    H = config["time"]["lookahead_weeks"]

    users = meta["users"]
    weeks_per_user = meta["weeks_per_user"]

    # --- choose a valid user if not specified ---
    if user_idx is None:
        valid_candidates = [
            u for u, weeks in enumerate(weeks_per_user)
            if len(weeks) >= W + 1
        ]
        if not valid_candidates:
            raise ValueError(f"No users with at least {W+1} weeks of history.")
        user_idx = int(np.random.choice(valid_candidates))

    weeks_u = weeks_per_user[user_idx]
    T_u = len(weeks_u)
    if T_u < W + 1:
        raise ValueError(f"User {user_idx} has only {T_u} weeks; need at least {W+1}.")

    # Slice per-user tensors to their valid length T_u
    E_u = E_pairs[user_idx, :T_u]       # [T_u, K, 2]
    X_u = X_week[user_idx, :T_u]        # [T_u, D_x]
    Y_u = Y[user_idx, :T_u]             # [T_u, K]
    M_u = M_target[user_idx, :T_u]      # [T_u, K]

    # replace Y_u nans with 0 for computation
    Y_u = np.nan_to_num(Y_u, nan=0.0)

    # --- Ground truth global score per week ---
    m_sum = M_u.sum(axis=1)  # [T_u]
    with np.errstate(invalid="ignore", divide="ignore"):
        g_truth = np.where(m_sum > 0, (Y_u * M_u).sum(axis=1) / m_sum, np.nan)

    # --- In-sample predictions (one-step ahead) ---
    pred_in_sample = np.full(T_u, np.nan, dtype=np.float32)

    Ds = 0 if statics is None else statics.shape[1]
    if Ds > 0:
        s_static_row = statics[user_idx].reshape(1, Ds)
        s_static_t = torch.from_numpy(s_static_row).float().to(device)
    else:
        s_static_t = torch.zeros((1, 0), dtype=torch.float32, device=device)

    Da = len(getattr(prescriptor, "action_names", []))

    for t in range(W, T_u):
        t0, t1 = t - W, t
        # reconstruct y_hist and x_hist for [t-W, ..., t-1]
        pres_win = E_u[t0:t1, :, 0]                    # [W, K]
        inv_win  = E_u[t0:t1, :, 1]
        y_hist_np = pres_win * (1.0 - inv_win)         # [W, K]

        E_flat_win = E_u[t0:t1].reshape(W, -1)         # [W, 2K]
        x_hist_np  = np.concatenate([E_flat_win, X_u[t0:t1, :]], axis=-1)  # [W, 2K + D_x]

        y_hist = torch.from_numpy(y_hist_np).unsqueeze(0).float().to(device)
        x_hist = torch.from_numpy(x_hist_np).unsqueeze(0).float().to(device)

        # action at week t: for now, zero (or derived from M_target if you want)
        if Da > 0:
            a_now_np = np.zeros((1, Da), dtype=np.float32)
            a_now = torch.from_numpy(a_now_np).float().to(device)
        else:
            a_now = torch.zeros((1, 0), dtype=torch.float32, device=device)

        user_idx_t = torch.tensor([user_idx], dtype=torch.long, device=device)

        with torch.no_grad():
            preds = model(y_hist, x_hist, a_now, s_static_t, user_idx_t)
            g_hat = float(preds["g_mu"].item())
        pred_in_sample[t] = g_hat

    # --- Unrolled future predictions (beyond T_u) ---
    # This uses your unroll_predictor_weekly helper
    pred_df_future, pred_scores_future, pred_actions_future = unroll_predictor_weekly(
        config=config,
        model=model,
        prescriptor=prescriptor,
        E_pairs=E_pairs,
        X_week=X_week,
        meta=meta,
        user_idx=user_idx,
        statics_row=None if statics is None else statics[user_idx],
        device=device,
    )

    # Expect a 'week_start' column in pred_df_future
    future_weeks = pd.to_datetime(pred_df_future["week_start"].values)
    g_future = pred_df_future["global_score"].to_numpy(dtype=float)

    # --- Plot ---
    plt.figure(figsize=(10, 5))

    # Ground truth
    plt.plot(weeks_u, g_truth, marker="o", linestyle="-", label="Ground truth (global, weekly)")

    # In-sample predictions
    plt.plot(weeks_u, pred_in_sample, marker="x", linestyle="--", label="Predictions (in-sample)")

    # Unrolled future
    plt.plot(future_weeks, g_future, marker="s", linestyle="-.", label="Unrolled predictions (future)")

    # Mark where ground truth ends
    plt.axvline(weeks_u[-1], color="k", linestyle=":", linewidth=1.5, label="End of ground truth")

    plt.title(f"User {users[user_idx]} — Weekly performance and predictions")
    plt.xlabel("Week")
    plt.ylabel("Global score (mean of domains)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
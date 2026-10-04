for _mode, _D in MODE_DIRS.items():
    print(f"\n===== hw_threshold_mode = {_mode} =====")
    # ---- 12b. Axis II strict null (SLOW: n_perm x [ECA + unique attribution]) ----
    # Permutations per mode: cfg.n_null_permutations_by_mode (calendar 1000,
    # annual 500); for a quick test set e.g. {"calendar": 200, "annual": 200}.
    SIG = _D / "significance"
    set_mode(_mode)
    axis2_null(cfg, _D, SIG)
    axis2_window_sensitivity(cfg, _D, SIG)
    check_axis2_consistency(cfg, _D, SIG)          # raises on any mismatch
    null = pd.read_parquet(SIG / "axis2_null_region.parquet")
    print(null[null.season == "ALL"][["scale", "n_hw", "count_ratio_obs", "count_ratio_null_med",
          "count_ratio_null_lo", "count_ratio_null_hi", "trigger_obs",
          "trigger_null_hi"]].round(4).to_string(index=False))   # p-values stored, not shown
    # decomposition: observed before/after counts vs their null medians
    print(null[null.season == "ALL"][["scale", "n_before_obs", "n_before_null_med",
          "before_obs_over_null_med", "n_after_obs", "n_after_null_med",
          "after_obs_over_null_med"]].round(4).to_string(index=False))
    print(pd.read_parquet(SIG / "axis2_window_sensitivity.parquet")
          .query("scale == 'Continental'")[["window_days", "n_hw", "count_ratio", "rate_diff"]]
          .round(4).to_string(index=False))

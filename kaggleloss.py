class KaggleLoss(nn.Module):
    def __init__(self, var_weights=None, metric_var_weights=None, eps=1e-6):
        super().__init__()
        # build and normalize cosine‐latitude weights
        ds = xr.open_zarr(config["data"]["path"], consolidated=False)
        lats = ds["latitude"].values if "latitude" in ds.coords else ds["y"].values
        cosl = np.cos(np.deg2rad(lats))
        cosl = cosl / cosl.sum()                 # sum to 1
        self.lat_w = torch.tensor(cosl, dtype=torch.float32)  # plain attr

        # default variable‐level weights
        self.var_weights = var_weights or {"tas":0.5,"pr":0.5}
        self.metric_var_weights = metric_var_weights or {
            "tas": {"monthly_rmse":0.1, "time_mean":1.0, "time_std":1.0},
            "pr" : {"monthly_rmse":0.1, "time_mean":1.0, "time_std":0.75},
        }
        self.eps = eps

    def forward(self, y_hat, y_true):
        """
        y_hat, y_true: (B, C, H, W)
        """
        device = y_hat.device
        lat_w = self.lat_w.to(device)
        B, C, H, W = y_hat.shape
        assert H == self.lat_w.numel(), "Height mismatch vs lat_w"
        # spatial weighting dims
        lat_w = lat_w.view(1, H, 1)
        lon_w = 1.0 / W

        total = 0.0
        for idx, var in enumerate(["tas","pr"]):
            p = y_hat[:,idx]    # (B,H,W)
            t = y_true[:,idx]   # (B,H,W)

            # 1) monthly RMSE
            sq = (p-t).pow(2)                     # (B,H,W)
            tavg = sq.mean(dim=0)                # (H,W)
            wavg = (tavg*lat_w).sum(dim=0)*lon_w
            monthly_rmse = torch.sqrt(wavg.mean() + self.eps)

            # 2) time‐mean RMSE
            mp = p.mean(dim=0);  mt = t.mean(dim=0)             # (H,W)
            sqm = (mp-mt).pow(2)
            wtm = (sqm*lat_w).sum(dim=0)*lon_w
            time_mean_rmse = torch.sqrt(wtm.mean() + self.eps)

            # 3) time‐std MAE
            sp = p.std(dim=0, unbiased=False); st = t.std(dim=0, unbiased=False)
            absd = (sp-st).abs()
            wstd = (absd*lat_w).sum(dim=0)*lon_w
            time_std_mae = wstd.mean()

            m = self.metric_var_weights[var]
            var_score = (m["monthly_rmse"]*monthly_rmse
                       + m["time_mean"]     *time_mean_rmse
                       + m["time_std"]      *time_std_mae)

            total += self.var_weights[var] * var_score

        return total
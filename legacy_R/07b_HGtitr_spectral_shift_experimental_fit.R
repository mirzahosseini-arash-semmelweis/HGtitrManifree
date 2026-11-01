################################################################################
## Measured example of full spectrum Host-Guest titration fit
################################################################################

source("01_HGtitr_simulator.R")
source("02_HGtitr_fitter.R")
source("03_HGtitr_uncertainty_estimator.R")
source("04_HGtitr_multichannel_fitter.R")
source("05_HGtitr_multichannel_uncertainty_estimator.R")
source("06_HGtitr_spectral_shift_fitter.R")
source("07_HGtitr_spectral_shift_uncertainty_estimator.R")
library(dplyr)
library(purrr)
library(tibble)
library(ggplot2)
library(gridExtra)
library(signs)

# Loading and filtering spectrum -----------------------------------------------

biliverdin_fullCD <- read.csv("biliverdin_rabsa_fullCD.csv")

wavelength_cols <- grep(
  "^CD",
  names(biliverdin_fullCD),
  value = TRUE
)

wavelength_values <- as.numeric(
  sub(
    "^CD",
    "",
    wavelength_cols
  )
)

bvd_long <- HG_spectra_wide_to_long(
  data              = biliverdin_fullCD,
  wavelength_cols   = wavelength_cols,
  wavelength_values = wavelength_values
)

ggplot(bvd_long) +
  geom_line(aes(x = wavelength, y = signal, color = factor(total_G))) +
  theme_bw()

ggplot(bvd_long |> filter(wavelength > 305 & wavelength < 445)) +
  geom_line(aes(x = wavelength, y = signal, color = factor(total_G))) +
  theme_bw()

bvd_long <- bvd_long |> filter(wavelength > 305 & wavelength < 445)

# Occupancy-dependent spectral shift model -------------------------------------

## 1:2 model (H1G1+H1G2) based on multichannel fit
### H1G1 spectral-shift
leaderboard_bvd_shift_st12_sp1 <- multistart_HG_spectral_shift(
  data_long       = bvd_long,
  species_vec     = c("H1G1", "H1G2"),
  logB_center     = c(5, 8),
  logB_lower      = c(2, 4),
  logB_upper      = c(9, 14),
  n_start         = 50,
  logB_jitter     = 1.5,
  shift_mode      = "occupancy_linear",
  shifted_species = c("H1G1"),
  delta_range     = c(-200, 200),
  spline_df       = 10,
  ridge           = 1e-8,
  seed            = 123,
  
  save_dir = "HG_fits_bvd_shift_st12_sp1",
  
  nloptr_control = list(
    maxeval     = 2500,
    xtol_rel    = 1e-6,
    print_level = 0
  )
)

leaderboard_bvd_shift_st12_sp1[1:5, ]
fit_bvd_shift_st12_sp1 <- readRDS(leaderboard_bvd_shift_st12_sp1[1, "file"])
fit_bvd_shift_st12_sp1$par # logB2 did not converge

### H1G1 spectral-shift
leaderboard_bvd_shift_st12_sp2 <- multistart_HG_spectral_shift(
  data_long       = bvd_long,
  species_vec     = c("H1G1", "H1G2"),
  logB_center     = c(5, 8),
  logB_lower      = c(2, 4),
  logB_upper      = c(9, 14),
  n_start         = 50,
  logB_jitter     = 1.5,
  shift_mode      = "occupancy_linear",
  shifted_species = c("H1G2"),
  delta_range     = c(-200, 200),
  spline_df       = 10,
  ridge           = 1e-8,
  seed            = 123,
  
  save_dir = "HG_fits_bvd_shift_st12_sp2",
  
  nloptr_control = list(
    maxeval     = 2500,
    xtol_rel    = 1e-6,
    print_level = 0
  )
)

leaderboard_bvd_shift_st12_sp2[1:5, ]
fit_bvd_shift_st12_sp2 <- readRDS(leaderboard_bvd_shift_st12_sp2[1, "file"])
fit_bvd_shift_st12_sp2$par # logB2 did not converge

### H1G1+H1G2 spectral-shift
leaderboard_bvd_shift_st12_sp12 <- multistart_HG_spectral_shift(
  data_long       = bvd_long,
  species_vec     = c("H1G1", "H1G2"),
  logB_center     = c(5.3, 9.5),
  logB_lower      = c(2, 4),
  logB_upper      = c(9, 14),
  n_start         = 50,
  logB_jitter     = 1.5,
  shift_mode      = "occupancy_linear",
  shifted_species = c("H1G1", "H1G2"),
  delta_range     = c(-200, 200),
  spline_df       = 10,
  ridge           = 1e-8,
  seed            = 123,
  
  save_dir = "HG_fits_bvd_shift_st12_sp12",
  
  nloptr_control = list(
    maxeval     = 2500,
    xtol_rel    = 1e-6,
    print_level = 0
  )
)

leaderboard_bvd_shift_st12_sp12[1:5, ]
fit_bvd_shift_st12_sp12 <- readRDS(leaderboard_bvd_shift_st12_sp12[1, "file"])
fit_bvd_shift_st12_sp12$par

## 1a:1b model (H1G1a+H1G1b) based on spectrum isosbestic point
### H1G1a spectral-shift
leaderboard_bvd_shift_st1a1b_sp1a <- multistart_HG_spectral_shift(
  data_long       = bvd_long,
  species_vec     = c("H1G1a", "H1G1b"),
  logB_center     = c(4.5, 6),
  logB_lower      = c(2, 4),
  logB_upper      = c(9, 14),
  n_start         = 50,
  logB_jitter     = 1.5,
  shift_mode      = "occupancy_linear",
  shifted_species = c("H1G1a"),
  delta_range     = c(-120, 120),
  spline_df       = 10,
  ridge           = 1e-8,
  seed            = 123,
  
  save_dir = "HG_fits_bvd_shift_st1a1b_sp1a",
  
  nloptr_control = list(
    maxeval     = 2500,
    xtol_rel    = 1e-6,
    print_level = 0
  )
)

leaderboard_bvd_shift_st1a1b_sp1a[1:5, ]
fit_bvd_shift_st1a1b_sp1a <- readRDS(leaderboard_bvd_shift_st1a1b_sp1a[1, "file"])
fit_bvd_shift_st1a1b_sp1a$par

### H1G1a+H1G1b spectral-shift
leaderboard_bvd_shift_st1a1b_sp1a1b <- multistart_HG_spectral_shift(
  data_long       = bvd_long,
  species_vec     = c("H1G1a", "H1G1b"),
  logB_center     = c(4.5, 6),
  logB_lower      = c(2, 4),
  logB_upper      = c(9, 14),
  n_start         = 50,
  logB_jitter     = 1.5,
  shift_mode      = "occupancy_linear",
  shifted_species = c("H1G1a", "H1G1b"),
  delta_range     = c(-120, 120),
  spline_df       = 10,
  ridge           = 1e-8,
  seed            = 123,
  
  save_dir = "HG_fits_bvd_shift_st1a1b_sp1a1b",
  
  nloptr_control = list(
    maxeval     = 2500,
    xtol_rel    = 1e-6,
    print_level = 0
  )
)

leaderboard_bvd_shift_st1a1b_sp1a1b[1:5, ]
fit_bvd_shift_st1a1b_sp1a1b <- readRDS(leaderboard_bvd_shift_st1a1b_sp1a1b[1, "file"])
fit_bvd_shift_st1a1b_sp1a1b$par

# Comparing spectral-shift models with AR(1) AIC

spectral_fits_bvd <- list(
  st12_sp12     = fit_bvd_shift_st12_sp12,
  st1a1b_sp1a   = fit_bvd_shift_st1a1b_sp1a,
  st1a1b_sp1a1b = fit_bvd_shift_st1a1b_sp1a1b
)

fit_comparison_bvd_AR1 <- purrr::imap_dfr(
  spectral_fits_bvd,
  ~ extract_spectral_fit_metrics_AR1(
    fit = .x,
    model_name = .y
  )
) |>
  dplyr::arrange(
    AIC_AR1
  ) |>
  dplyr::mutate(
    
    delta_AIC_AR1   = AIC_AR1 - min(AIC_AR1),
    delta_AICc_AR1  = AICc_AR1_approx - min(AICc_AR1_approx),
    delta_BIC_AR1   = BIC_AR1 - min(BIC_AR1),
    AIC_weight_AR1  = exp(-0.5 * delta_AIC_AR1) / sum(exp(-0.5 * delta_AIC_AR1)),
    AICc_weight_AR1 = exp(-0.5 * delta_AICc_AR1) / sum(exp(-0.5 * delta_AICc_AR1))
  )

fit_comparison_bvd_AR1 |>
  dplyr::select(
    model,
    SSE,
    rho_AR1,
    sigma_AR1,
    logLik_AR1,
    AIC_AR1,
    delta_AIC_AR1,
    AIC_weight_AR1,
    AICc_AR1_approx,
    delta_AICc_AR1,
    BIC_AR1
  )
# model               SSE rho_AR1 sigma_AR1 logLik_AR1  AIC_AR1 delta_AIC_AR1 AIC_weight_AR1
# <chr>             <dbl>   <dbl>     <dbl>      <dbl>    <dbl>         <dbl>          <dbl>
# 1 st1a1b_sp1a1b    874.   0.999     0.215     56212. -112334.           0        1.000e+ 0
# 2 st1a1b_sp1a     1000.   0.999     0.232     56192. -112297.          37.5      7.23 e- 9
# 3 st12_sp12        905.   0.999     0.257     56138. -112184.         150.       3.07 e-33

# Bayesian MCMC uncertainty estimation

HG_spectral_estimate_rho(
  fit_bvd_shift_st1a1b_sp1a1b,
  clamp = c(-0.999999, 0.999999)
)
HG_spectral_estimate_sigma(
  fit_bvd_shift_st1a1b_sp1a1b
)
HG_spectral_supported_shift(
  fit_bvd_shift_st1a1b_sp1a1b
)

bounds_bvd <- make_HG_spectral_shift_bayes_bounds(
  fit = fit_bvd_shift_st1a1b_sp1a1b,
  
  noise_model = "ar1",
  
  lower_override = c(
    logB_H1G1a = 2,
    logB_H1G1b = 2,
    delta_lambda_H1G1a = -120,
    delta_lambda_H1G1b = -120
  ),
  
  upper_override = c(
    logB_H1G1a = 8,
    logB_H1G1b = 8,
    delta_lambda_H1G1a = 120,
    delta_lambda_H1G1b = 120
  )
)

# Sanity check
cbind(
  start = bounds_bvd$start,
  lower = bounds_bvd$lower,
  upper = bounds_bvd$upper
)

fit_bvd_shift_bayes <- fit_HG_spectral_shift_MCMC(
  fit = fit_bvd_shift_st1a1b_sp1a1b,
  
  noise_model = "ar1",
  
  lower_override = c(
    logB_H1G1a = 2,
    logB_H1G1b = 2,
    delta_lambda_H1G1a = -120,
    delta_lambda_H1G1b = -120
  ),
  
  upper_override = c(
    logB_H1G1a = 8,
    logB_H1G1b = 8,
    delta_lambda_H1G1a = 120,
    delta_lambda_H1G1b = 120
  ),
  
  iterations = 6000,
  burnin     = 1000,
  
  n_internal_chains = 6,
  
  seed = 0,
  
  consoleUpdates = 500
)

fit_bvd_shift_bayes$posterior_summary

# parameter                   mean     median           sd       lower      upper
# 1         logB_H1G1a   3.6335979  3.6333703 3.348737e-02   3.5481924  3.7096834
# 2         logB_H1G1b   5.7124615  5.7272542 6.449650e-02   5.5904256  5.8247846
# 3 delta_lambda_H1G1a  48.7658871 48.7589347 9.126068e-01  46.8210859 50.4282112
# 4 delta_lambda_H1G1b -15.3134450 -4.5992790 1.606217e+01 -40.0916433 -3.4976078
# 5          log_sigma  -1.6099271 -1.6681138 1.238200e-01  -1.7803799 -1.4099166
# 6         rho_lambda   0.9984601  0.9983217 4.267727e-04   0.9979155  0.9989891
# 7              sigma   0.2014699  0.1886025 2.566300e-02   0.1685741  0.2441637

BayesianTools::tracePlot(
  fit_bvd_shift_bayes$mcmc
)

BayesianTools::marginalPlot(
  fit_bvd_shift_bayes$mcmc
)

BayesianTools::correlationPlot(
  fit_bvd_shift_bayes$mcmc
)

# Extract plots

plot_spectrum <- ggplot(
  fit_bvd_shift_st1a1b_sp1a1b$prediction,
  aes(
    x     = wavelength,
    y     = signal,
    group = factor(point_id),
    color = factor(point_id)
  )
) +
  geom_line(
    alpha = 0.4
  ) +
  geom_line(
    aes(y = predicted),
    linewidth = 0.4
  ) +
  scale_colour_viridis_d(
    name   = "Titration point",
    option = "turbo",
    guide  = guide_legend(ncol = 2)
  ) +
  scale_y_continuous(labels = signs_format()) +
  labs(
    title    = "RabSA biliverdin titration",
    subtitle = "Spectral-shift fit with B-splines",
    y = "Ellipticity (mdeg)",
    x = "Wavelength (nm)"
  ) +
  theme_bw() +
  theme(
    plot.title    = element_text(hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
  )

wavelength_labels <- c(
  "383.5" = "383.5 nm",
  "400.7" = "400.7 nm",
  "410.9" = "410.9 nm",
  "437.6" = "437.6 nm"
)


label_scientific <- function(x) {
  ifelse(x == 0, "0", {
    text <- scales::scientific_format()(x)
    text <- gsub("1e", "10^", text)
    text <- gsub("e", " %*% 10^", text)
    text <- gsub("\\+0|\\+", "", text)
    text <- gsub("-0", "-", text)
    parse(text = text)
  })
}

plot_spectral_fit <- fit_bvd_shift_st1a1b_sp1a1b$prediction |>
  filter(wavelength %in% c(383.5, 400.7, 410.9, 437.6)) |>
  ggplot(aes(x = total_G, color = factor(wavelength))) +
  geom_point(
    aes(y = signal),
    shape = 1
  ) +
  geom_line(
    aes(y = predicted)
  ) +
  facet_wrap(
    ~ wavelength,
    scales   = "free",
    labeller = labeller(wavelength = wavelength_labels)
  ) +
  scale_color_manual(
    values = c(
      "383.5" = "#004488",
      "400.7" = "#EE7733",
      "410.9" = "#00C19F",
      "437.6" = "#EE3377"
    ),
    guide  = "none"
  ) +
  scale_x_continuous(
    breaks = c(0, 2e-5, 4e-5),
    labels = label_scientific
  ) +
  scale_y_continuous(labels = signs_format()) +
  labs(
    x = expression("[G]"[total]),
    y = "Induced circular dichroism signal"
  ) +
  theme_bw() +
  theme(
    strip.background = element_blank()
  )

plot_spectral_shift <- grid.arrange(
  plot_spectrum,
  plot_spectral_fit,
  nrow = 1
)

ggsave(
  "plot_spectral_shift.svg",
  plot_spectral_shift,
  width  = 12,
  height = 7
)
ggsave(
  "plot_spectral_shift.pdf",
  plot_spectral_shift,
  width  = 12,
  height = 7
)

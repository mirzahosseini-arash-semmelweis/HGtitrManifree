################################################################################
## Measured examples of Host-Guest titration fit
################################################################################

source("01_HGtitr_simulator.R")
source("02_HGtitr_fitter.R")
source("03_HGtitr_uncertainty_estimator.R")
source("04_HGtitr_multichannel_fitter.R")
source("05_HGtitr_multichannel_uncertainty_estimator.R")
library(dplyr)
library(purrr)
library(ggplot2)
library(tibble)

## Example 1: meloxicam - rabbit serum albumin titration with fluorescence quenching

meloxicam_rabsa_data <- read.csv("meloxicam_rabsa.csv")

fixed_params_mel <- list(
  # species_vec        = c("H1G1"),
  species_vec        = c("H1G1", "H1G2"), # try both 1:1 and 1:2 stoichiometry
  endpoint_tol       = 1e-7,
  endpoint_maxiter   = 500L,
  endpoint_damping   = 0.5,
  use_optim_fallback = TRUE
)

fit_spec_mel <- list( # set here for 1:2
  
  logB_vec = list(
    start     = c(5, 9),
    lower     = c(-Inf, -Inf),
    upper     = c( Inf, Inf),
    fit       = c(TRUE, TRUE),
    map_scale = c(1, 1)
  ),
  
  response_vec = list(
    
    y_bsFL282.9 = list(
      start     = c(7e4,  0,  -1e3, -6e3),
      lower     = c(1e4,  0,  -5e3, -5e4),
      upper     = c(1e5,  0,  8e3, 0),
      fit       = c(TRUE, FALSE, TRUE, TRUE),
      map_scale = c(1e4, NA, 1e3, 1e3)
    ),
    
    y_bsFL283 = list(
      start     = c(7e4,  0,  -1e3, -6e3),
      lower     = c(1e4,  0,  -5e3, -5e4),
      upper     = c(1e5,  0,  8e3, 0),
      fit       = c(TRUE, FALSE, TRUE, TRUE),
      map_scale = c(1e4, NA, 1e3, 1e3)
    )
  )
)

fit_mel <- fit_HG_multichannel_titration(
  fit_data     = meloxicam_rabsa_data,
  fixed_params = fixed_params_mel,
  fit_spec     = fit_spec_mel,
  
  signal_cols = c(
    y_bsFL282.9 = "y_bsFL282.9",
    y_bsFL283   = "y_bsFL283"
  ),
  
  channel_scale = "sd",
  
  method           = "nloptr",
  nloptr_algorithm = "NLOPT_LN_BOBYQA",
  loss             = "sse",
  
  nloptr_control = list(
    maxeval     = 1000,
    xtol_rel    = 1e-5,
    print_level = 1
  ),
  
  verbose = TRUE
)

fit_mel$logB_vec
fit_mel$response_vec
fit_mel$SSE_by_channel
fit_mel$RMSE_by_channel


ggplot(
  fit_mel$prediction_long,
  aes(x = total_G)
) +
  geom_point(
    aes(y = observed),
    shape = 1
  ) +
  geom_line(
    aes(y = predicted)
  ) +
  facet_wrap(
    ~ channel,
    scales = "free_y"
  ) +
  theme_bw()

ggplot(
  fit_mel$prediction_long,
  aes(
    x = total_G,
    y = residual
  )
) +
  geom_hline(
    yintercept = 0,
    linetype = 2
  ) +
  geom_point(
    shape = 1
  ) +
  geom_line() +
  facet_wrap(
    ~ channel,
    scales = "free_y"
  ) +
  theme_bw()

## Applying BayesianMCMC on competitive models

model_spec_mel <- tribble(
  ~model,    ~species_vec,               ~logB_start,  ~response_start,
  
  "model1",   c("H1G1"),                  c(5),         c(7e4, 0, -6e3),
  
  "model2",   c("H1G1", "H1G2"),          c(5, 9),      c(7e4, 0, 5e3, -2e4)
)

make_HG_model_fit_spec <- function(
    species_vec,
    logB_start,
    response_start,
    logB_map_scale = 1
) {
  
  n_species <- length(species_vec)
  
  stopifnot(
    length(logB_start) == n_species,
    length(response_start) == n_species + 2L
  )
  
  list(
    
    logB_vec = list(
      start     = logB_start,
      lower     = rep(-Inf, n_species),
      upper     = rep( Inf, n_species),
      fit       = rep(TRUE, n_species),
      map_scale = rep(logB_map_scale, n_species)
    ),
    
    response_vec = list(
      
      y_bsFL282.9 = list(
        start = response_start,
        
        lower = c(
          -Inf, 0,
          rep(-Inf, n_species)
        ),
        
        upper = c(
          Inf, 0,
          rep(Inf, n_species)
        ),
        
        fit = c(
          TRUE, FALSE,
          rep(TRUE, n_species)
        ),
        
        map_scale = c(
          abs(response_start[1]), NA,
          abs(response_start[-c(1:2)])
        )
      ),
      
      y_bsFL283 = list(
        start = response_start,
        
        lower = c(
          -Inf, 0,
          rep(-Inf, n_species)
        ),
        
        upper = c(
          Inf, 0,
          rep(Inf, n_species)
        ),
        
        fit = c(
          TRUE, FALSE,
          rep(TRUE, n_species)
        ),
        
        map_scale = c(
          abs(response_start[1]), NA,
          abs(response_start[-c(1:2)])
        )
      )
    )
  )
}

model_spec_mel <- model_spec_mel |>
  mutate(
    fit_spec = pmap(
      list(
        species_vec,
        logB_start,
        response_start
      ),
      make_HG_model_fit_spec
    )
  )

fit_grid_mel <- tidyr::crossing(
  model = model_spec_mel$model
)

fit_grid_mel <- fit_grid_mel |>
  mutate(
    fit = map(
      model,
      function(model_name) {
        
        model_row <- model_spec_mel |>
          filter(model == model_name)
        
        fixed_params <- fixed_params_mel
        
        fixed_params$species_vec <- model_row$species_vec[[1]]
        
        ## ----------------------------------------------------------
        ## Fit
        ## ----------------------------------------------------------
        
        fit_HG_multichannel_nloptr_MCMC(
          fit_data     = meloxicam_rabsa_data,
          fixed_params = fixed_params,
          fit_spec     = model_row$fit_spec[[1]],
          
          signal_cols = c(
            y_bsFL282.9 = "y_bsFL282.9",
            y_bsFL283   = "y_bsFL283"
          ),
          
          fitter_args = list(
            channel_scale = "mad",
            
            method           = "nloptr",
            nloptr_algorithm = "NLOPT_LN_BOBYQA",
            loss             = "sse",
            
            nloptr_control = list(
              maxeval     = 1000,
              xtol_rel    = 1e-5,
              print_level = 1
            ),
            
            verbose = TRUE
          ),
          
          mcmc_args = list(
            iterations = 4000L,
            burnin     = 1000L,
            n_internal_chains = 4L,
            n_Z = 100L,
            seed = 1
          )
        )
      }
    )
  )


dir.create(
  "HG_fits_meloxicam_rabsa",
  showWarnings = FALSE
)

pwalk(
  fit_grid_mel,
  function(
    model,
    fit
  ) {
    
    saveRDS(
      fit,
      file = file.path(
        "HG_fits_meloxicam_rabsa",
        paste0(
          model,
          ".rds"
        )
      )
    )
  }
)

fits_mel <- setNames(
  fit_grid_mel$fit,
  fit_grid_mel$model
)

fit_summary_mel <- fit_grid_mel |>
  mutate(
    objective = map_dbl(
      fit,
      ~ .x$fit$objective_value
    ),
    
    npar = map_int(
      fit,
      ~ length(.x$fit$par)
    ),
    
    convergence = map_int(
      fit,
      ~ .x$fit$fit$status
    )
  ) |>
  select(
    model,
    npar,
    objective,
    convergence
  )

fit_summary_mel

fit_parameters_mel <- fit_grid_mel |>
  transmute(
    model,
    
    parameter_table = map(
      fit,
      function(x) {
        
        det <- x$fit
        
        det$par_table |>
          mutate(
            estimate = unname(
              det$par[name]
            )
          ) |>
          select(
            name,
            group,
            channel,
            species,
            estimate,
            start,
            lower,
            upper
          )
      }
    )
  ) |>
  tidyr::unnest(
    parameter_table
  )

fit_parameters_mel

posterior_summary_mel <- fit_grid_mel |>
  transmute(
    model,
    
    posterior_summary = map(
      fit,
      ~ .x$posterior_summary
    )
  ) |>
  tidyr::unnest(
    posterior_summary
  )

posterior_summary_mel

posterior_logB_mel <- posterior_summary_mel |>
  filter(
    grepl("^logB_", parameter)
  )

posterior_logB_mel

posterior_response_mel <- posterior_summary_mel |>
  filter(
    grepl("^response_", parameter)
  )

fit_predictions_mel <- fit_grid_mel |>
  transmute(
    model,
    
    prediction = map(
      fit,
      ~ .x$fit$prediction_long
    )
  ) |>
  tidyr::unnest(
    prediction
  )

fit_predictions_mel

channel_fit_metrics_mel <- fit_predictions_mel |>
  group_by(
    model,
    channel
  ) |>
  summarise(
    n = n(),
    
    SSE = sum(residual^2),
    
    RMSE = sqrt(mean(residual^2)),
    
    MAE = mean(abs(residual)),
    
    .groups = "drop"
  )

channel_fit_metrics_mel

model_comparison_mel <- fit_grid_mel |>
  mutate(
    metrics = map(
      fit,
      extract_HG_MC_model_metrics
    )
  ) |>
  select(
    model,
    metrics
  ) |>
  tidyr::unnest(
    metrics
  ) |>
  mutate(
    delta_AICc_fixed = AICc_fixed - min(AICc_fixed),
    
    delta_AICc_profile = AICc_profile - min(AICc_profile),
    
    weight_AICc_fixed = {
      w <- exp(-0.5 * delta_AICc_fixed)
      
      w / sum(w)
    },
    
    weight_AICc_profile = {
      w <- exp(-0.5 * delta_AICc_profile)
      
      w / sum(w)
    }
  ) |>
  arrange(
    AICc_profile
  )

model_comparison_mel |>
  select(
    model,
    N,
    p,
    SSE,
    AICc_fixed,
    delta_AICc_fixed,
    weight_AICc_fixed,
    AICc_profile,
    delta_AICc_profile,
    weight_AICc_profile
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

channel_labels <- c(
  "y_bsFL282.9" = "282.9 nm",
  "y_bsFL283"   = "283.0 nm"
)

model_labels <- c(
  "model1" = "1:1 binding",
  "model2" = "1:2 binding"
)

plot_fit_mel <- ggplot(
  fit_predictions_mel,
  aes(x = total_G, color = channel, linetype = model)
) +
  geom_point(
    aes(y = observed),
    shape = 1,
    size  = 1.5,
    show.legend = FALSE
  ) +
  geom_line(
    aes(y = predicted),
    linewidth = 0.6
  ) +
  facet_grid(
    channel ~ model,
    scales   = "free_y",
    labeller = labeller(channel = channel_labels, model = model_labels)
  ) +
  scale_color_manual(
    values = c("y_bsFL282.9" = "#00BA38", "y_bsFL283" = "#FF61C3"),
    guide  = "none"
  ) +
  scale_linetype_manual(
    values = c("model1" = "solid", "model2" = "dashed"),
    guide  = "none"
  ) +
  scale_x_continuous(
    breaks = c(0, 3e-5, 6e-5, 9e-5),
    labels = label_scientific
  ) +
  labs(
    x = expression("[G]"[total]),
    y = "Normalized total fluorescence intensity"
  ) +
  theme_bw() +
  theme(
    strip.background = element_blank(),
    axis.text.x      = element_text(angle = 45, vjust = 0.5)
  )

plot_fit_mel

aic_labels_mel <- fit_predictions_mel |>
  group_by(
    model,
    channel
  ) |>
  summarise(
    x_min = min(total_G, na.rm = TRUE),
    
    x_max = max(total_G, na.rm = TRUE),
    
    .groups = "drop"
  ) |>
  mutate(
    x = (x_min + x_max) / 2
  ) |>
  left_join(
    model_comparison_mel |>
      select(
        model,
        delta_AICc_profile
      ),
    by = "model"
  ) |>
  mutate(
    label = sprintf(
      "\u0394AICc = %.2f",
      delta_AICc_profile
    )
  )

sigma_mel <- fit_grid_mel |>
  transmute(
    model,
    sigma = map(
      fit,
      function(x) {
        
        x$posterior_summary |>
          filter(
            grepl("^sigma_", parameter)
          ) |>
          transmute(
            channel_safe = sub(
              "^sigma_",
              "",
              parameter
            ),
            sigma_mean  = mean,
            sigma_lower = lower,
            sigma_upper = upper
          )
      }
    )
  ) |>
  unnest(sigma) |>
  mutate(
    channel = case_when(
      channel_safe == "y_bsFL282_9" ~ "y_bsFL282.9",
      channel_safe == "y_bsFL283"   ~ "y_bsFL283",
      TRUE                          ~ channel_safe
    )
  )

plot_residuals_mel <- ggplot(
  fit_predictions_mel,
  aes(x = total_G, y = residual, color = channel, linetype = model)
) +
  geom_rect(
    data = sigma_mel,
    aes(
      xmin = -Inf,
      xmax = Inf,
      ymin = -2 * sigma_mean,
      ymax =  2 * sigma_mean
    ),
    inherit.aes = FALSE,
    alpha       = 0.08
  ) +
  geom_hline(
    yintercept = 0,
    linewidth  = 0.35
  ) +
  geom_point(
    shape = 1,
    size  = 1.5
  ) +
  geom_line(
    linewidth = 0.6
  ) +
  geom_text(
    data = aic_labels_mel,
    aes(x = x, y = Inf, label = label),
    inherit.aes = FALSE,
    hjust = 0.5,
    vjust = 1.3,
    size  = 3
  ) +
  facet_grid(
    channel ~ model,
    scales   = "free_y",
    labeller = labeller(channel = channel_labels, model = model_labels)
  ) +
  scale_color_manual(
    values = c("y_bsFL282.9" = "#00BA38", "y_bsFL283" = "#FF61C3"),
    guide  = "none"
  ) +
  scale_linetype_manual(
    values = c("model1" = "solid", "model2" = "dashed"),
    guide  = "none"
  ) +
  scale_x_continuous(
    breaks = c(0, 3e-5, 6e-5, 9e-5),
    labels = label_scientific
  ) +
  scale_y_continuous(
    limits = c(-0.05, 0.05),
    labels = signs_format()
  ) +
  labs(
    x = expression("[G]"[total]),
    y = "Residual"
  ) +
  theme_bw() +
  theme(
    strip.background = element_blank(),
    axis.text.x      = element_text(angle = 45, vjust = 0.5)
  )

plot_residuals_mel

plot_mel <- grid.arrange(
  plot_fit_mel,
  plot_residuals_mel,
  ncol = 1
)

ggsave(
  "plot_mel.svg",
  plot_mel,
  width  = 6,
  height = 8
)

## Example 2: biliverdin - rabbit serum albumin titration with fluorescence quenching

biliverdin_rabsa_data <- read.csv("biliverdin_rabsa.csv")

# exploring 1:1 model

fixed_params_bvd <- list(
  species_vec        = c("H1G1"),
  endpoint_tol       = 1e-7,
  endpoint_maxiter   = 500L,
  endpoint_damping   = 0.5,
  use_optim_fallback = TRUE
)

fit_spec_bvd <- list(
  
  logB_vec = list(
    start     = c(6),
    lower     = c(-Inf),
    upper     = c( Inf),
    fit       = c(TRUE),
    map_scale = c(1)
  ),
  
  response_vec = list(
    CD400.7 = list(
      start     = c( 2e3,  0,  -4e5),
      lower     = c(-Inf,  0,  -Inf),
      upper     = c( Inf,  0,   Inf),
      fit       = c(TRUE, FALSE, TRUE),
      map_scale = c(1e3, NA, 1e5)
    ),
    CD383.5 = list(
      start     = c( 7e2,  0,  -5e5),
      lower     = c(-Inf,  0,  -Inf),
      upper     = c( Inf,  0,   Inf),
      fit       = c(TRUE, FALSE, TRUE),
      map_scale = c(1e3, NA, 1e5)
    ),
    CD410.9 = list(
      start     = c(-9e2,  0,  -3e5),
      lower     = c(-Inf,  0,  -Inf),
      upper     = c( Inf,  0,   Inf),
      fit       = c(TRUE, FALSE, TRUE),
      map_scale = c(1e3, NA, 1e5)
    ),
    CD437.6 = list(
      start     = c(-9e2,  0,  -1e5),
      lower     = c(-Inf,  0,  -Inf),
      upper     = c( Inf,  0,   Inf),
      fit       = c(TRUE, FALSE, TRUE),
      map_scale = c(1e3, NA, 1e5)
    )
  )
)

fit_bvd <- fit_HG_multichannel_titration(
  fit_data     = biliverdin_rabsa_data,
  fixed_params = fixed_params_bvd,
  fit_spec     = fit_spec_bvd,
  
  signal_cols = c(
    CD400.7 = "CD400.7",
    CD383.5 = "CD383.5",
    CD410.9 = "CD410.9",
    CD437.6 = "CD437.6"
  ),
  
  channel_scale = "mad",
  
  method           = "nloptr",
  nloptr_algorithm = "NLOPT_LN_BOBYQA",
  loss             = "sse",
  
  nloptr_control = list(
    maxeval     = 1000,
    xtol_rel    = 1e-5,
    print_level = 1
  ),
  
  verbose = TRUE
)

fit_bvd$logB_vec
fit_bvd$response_vec

ggplot(
  fit_bvd$prediction_long,
  aes(x = total_G)
) +
  geom_point(
    aes(y = observed),
    shape = 1
  ) +
  geom_line(
    aes(y = predicted)
  ) +
  facet_wrap(
    ~ channel,
    scales = "free_y"
  ) +
  theme_bw()

# exploring 1:2 model

fixed_params_bvd <- list(
  species_vec        = c("H1G1", "H1G2"),
  endpoint_tol       = 1e-7,
  endpoint_maxiter   = 500L,
  endpoint_damping   = 0.5,
  use_optim_fallback = TRUE
)

fit_spec_bvd <- list(
  
  logB_vec = list(
    start     = c(5.4, 10.1),
    lower     = c(-Inf, -Inf),
    upper     = c( Inf,  Inf),
    fit       = c(TRUE, TRUE),
    map_scale = c(1, 2)
  ),
  
  response_vec = list(
    CD400.7 = list(
      start     = c( 2.6e3,  0,  -6.7e5, -2.6e5),
      lower     = c(-Inf,  0,  -Inf, -Inf),
      upper     = c( Inf,  0,   Inf,  Inf),
      fit       = c(TRUE, FALSE, TRUE, TRUE),
      map_scale = c(1e3, NA, 1e5, 1e5)
    ),
    CD383.5 = list(
      start     = c( 7.4e2,  0,  -5.25e5, -1.3e6),
      lower     = c(-Inf,  0,  -Inf, -Inf),
      upper     = c( Inf,  0,   Inf,  Inf),
      fit       = c(TRUE, FALSE, TRUE, TRUE),
      map_scale = c(1e2, NA, 1e5, 1e6)
    ),
    CD410.9 = list(
      start     = c(-6.95e2,  0,  -7.3e5, 3.9e3),
      lower     = c(-Inf,  0,  -Inf, -Inf),
      upper     = c( Inf,  0,   Inf,  Inf),
      fit       = c(TRUE, FALSE, TRUE, TRUE),
      map_scale = c(1e2, NA, 1e5, 1e4)
    ),
    CD437.6 = list(
      start     = c(-9.3e2,  0,  -2.8e5, 3.3e5),
      lower     = c(-Inf,  0,  -Inf, -Inf),
      upper     = c( Inf,  0,   Inf,  Inf),
      fit       = c(TRUE, FALSE, TRUE, TRUE),
      map_scale = c(1e2, NA, 1e5, 1e5)
    )
  )
)

fit_bvd <- fit_HG_multichannel_titration(
  fit_data     = biliverdin_rabsa_data,
  fixed_params = fixed_params_bvd,
  fit_spec     = fit_spec_bvd,
  
  signal_cols = c(
    CD400.7 = "CD400.7",
    CD383.5 = "CD383.5",
    CD410.9 = "CD410.9",
    CD437.6 = "CD437.6"
  ),
  
  channel_scale = "mad",
  
  method           = "nloptr",
  nloptr_algorithm = "NLOPT_LN_BOBYQA",
  loss             = "sse",
  
  nloptr_control = list(
    maxeval     = 10000,
    xtol_rel    = 1e-6,
    print_level = 1
  ),
  
  verbose = TRUE
)

fit_bvd$logB_vec
fit_bvd$response_vec

ggplot(
  fit_bvd$prediction_long,
  aes(x = total_G)
) +
  geom_point(
    aes(y = observed),
    shape = 1
  ) +
  geom_line(
    aes(y = predicted)
  ) +
  facet_wrap(
    ~ channel,
    scales = "free_y"
  ) +
  theme_bw()

# > fit_bvd$logB_vec
# [1]  5.447015 10.237214
# > fit_bvd$response_vec
# $CD400.7
# [1]    2340.07       0.00 -669188.21 -268153.76
# 
# $CD383.5
# [1]      746.0065        0.0000  -493095.4609 -1297129.2622
# 
# $CD410.9
# [1]    -687.5352       0.0000 -727780.9883  336951.7392
# 
# $CD437.6
# [1]    -936.6557       0.0000 -278530.3147  296865.0449

# exploring 1:3 model

fixed_params_bvd <- list(
  species_vec        = c("H1G1", "H1G2", "H1G3"),
  endpoint_tol       = 1e-7,
  endpoint_maxiter   = 500L,
  endpoint_damping   = 0.5,
  use_optim_fallback = TRUE
)

fit_spec_bvd <- list(
  
  logB_vec = list(
    start     = c(6.97, 8.3, 19.97),
    lower     = c(-Inf, -Inf, -Inf),
    upper     = c( Inf,  Inf, Inf),
    fit       = c(TRUE, TRUE, TRUE),
    map_scale = c(1, 2, 3)
  ),
  
  response_vec = list(
    CD400.7 = list(
      start     = c( 1.7e3,  0,  -5.8e5, -3.7e5, -4.6e5),
      lower     = c(-Inf,  0,  -Inf, -Inf, -Inf),
      upper     = c( Inf,  0,   Inf,  Inf, Inf),
      fit       = c(TRUE, FALSE, TRUE, TRUE, TRUE),
      map_scale = c(1e3, NA, 1e5, 1e5, 1e5)
    ),
    CD383.5 = list(
      start     = c( 9.8e2,  0,  -4.6e5, -8.2e5, -1.36e6),
      lower     = c(-Inf,  0,  -Inf, -Inf, -Inf),
      upper     = c( Inf,  0,   Inf,  Inf, Inf),
      fit       = c(TRUE, FALSE, TRUE, TRUE, TRUE),
      map_scale = c(1e3, NA, 1e5, 1e5, 1e6)
    ),
    CD410.9 = list(
      start     = c(9.5e3,  0,  -6.0e5, 5.1e5, 2.4e4),
      lower     = c(-Inf,  0,  -Inf, -Inf, -Inf),
      upper     = c( Inf,  0,   Inf,  Inf, Inf),
      fit       = c(TRUE, FALSE, TRUE, TRUE, TRUE),
      map_scale = c(1e3, NA, 1e5, 1e5, 1e5)
    ),
    CD437.6 = list(
      start     = c(-8.5e2,  0,  -2.2e5, 6.4e3, 1.5e5),
      lower     = c(-Inf,  0,  -Inf, -Inf, -Inf),
      upper     = c( Inf,  0,   Inf,  Inf, Inf),
      fit       = c(TRUE, FALSE, TRUE, TRUE, TRUE),
      map_scale = c(1e3, NA, 1e5, 1e5, 1e5)
    )
  )
)

fit_bvd <- fit_HG_multichannel_titration(
  fit_data     = biliverdin_rabsa_data,
  fixed_params = fixed_params_bvd,
  fit_spec     = fit_spec_bvd,
  
  signal_cols = c(
    CD400.7 = "CD400.7",
    CD383.5 = "CD383.5",
    CD410.9 = "CD410.9",
    CD437.6 = "CD437.6"
  ),
  
  channel_scale = "mad",
  
  method           = "nloptr",
  nloptr_algorithm = "NLOPT_LN_BOBYQA",
  loss             = "sse",
  
  nloptr_control = list(
    maxeval     = 10000,
    xtol_rel    = 1e-6,
    print_level = 1
  ),
  
  verbose = TRUE
)

fit_bvd$logB_vec
fit_bvd$response_vec

ggplot(
  fit_bvd$prediction_long,
  aes(x = total_G)
) +
  geom_point(
    aes(y = observed),
    shape = 1
  ) +
  geom_line(
    aes(y = predicted)
  ) +
  facet_wrap(
    ~ channel,
    scales = "free_y"
  ) +
  theme_bw()

# > fit_bvd$logB_vec
# [1]  7.493448  8.581465 21.560306
# > fit_bvd$response_vec
# $CD400.7
# [1]    2259.611       0.000 -574439.818 -379145.905 -468174.889
# 
# $CD383.5
# [1]     1160.079        0.000  -457259.095  -823786.360 -1353474.045
# 
# $CD410.9
# [1]    9628.404       0.000 -600416.614  509389.490   17902.115
# 
# $CD437.6
# [1]     163.562       0.000 -218725.744    4541.109  148677.189

# exploring 1:1, 2:1, 1:3 model

fixed_params_bvd <- list(
  species_vec        = c("H1G1", "H1G2", "H1G3", "H2G1"),
  endpoint_tol       = 1e-7,
  endpoint_maxiter   = 500L,
  endpoint_damping   = 0.5,
  use_optim_fallback = TRUE
)

fit_spec_bvd <- list(
  
  logB_vec = list(
    start     = c(6.97, 8.3, 19.97, 9),
    lower     = c(-Inf, -Inf, -Inf, -Inf),
    upper     = c( Inf,  Inf, Inf, Inf),
    fit       = c(TRUE, TRUE, TRUE, TRUE),
    map_scale = c(1, 2, 3, 3)
  ),
  
  response_vec = list(
    CD400.7 = list(
      start     = c( 1.7e3,  0,  -5.8e5, -3.7e5, -4.6e5, 0),
      lower     = c(-Inf,  0,  -Inf, -Inf, -Inf, -Inf),
      upper     = c( Inf,  0,   Inf,  Inf, Inf, Inf),
      fit       = c(TRUE, FALSE, TRUE, TRUE, TRUE, TRUE),
      map_scale = c(1e3, NA, 1e5, 1e5, 1e5, 1e5)
    ),
    CD383.5 = list(
      start     = c( 9.8e2,  0,  -4.6e5, -8.2e5, -1.36e6, 0),
      lower     = c(-Inf,  0,  -Inf, -Inf, -Inf, -Inf),
      upper     = c( Inf,  0,   Inf,  Inf, Inf, Inf),
      fit       = c(TRUE, FALSE, TRUE, TRUE, TRUE, TRUE),
      map_scale = c(1e3, NA, 1e5, 1e5, 1e6, 1e5)
    ),
    CD410.9 = list(
      start     = c(9.5e3,  0,  -6.0e5, 5.1e5, 2.4e4, 0),
      lower     = c(-Inf,  0,  -Inf, -Inf, -Inf, -Inf),
      upper     = c( Inf,  0,   Inf,  Inf, Inf, Inf),
      fit       = c(TRUE, FALSE, TRUE, TRUE, TRUE, TRUE),
      map_scale = c(1e3, NA, 1e5, 1e5, 1e5, 1e5)
    ),
    CD437.6 = list(
      start     = c(-8.5e2,  0,  -2.2e5, 6.4e3, 1.5e5, 0),
      lower     = c(-Inf,  0,  -Inf, -Inf, -Inf, -Inf),
      upper     = c( Inf,  0,   Inf,  Inf, Inf, Inf),
      fit       = c(TRUE, FALSE, TRUE, TRUE, TRUE, TRUE),
      map_scale = c(1e3, NA, 1e5, 1e5, 1e5, 1e5)
    )
  )
)

fit_bvd <- fit_HG_multichannel_titration(
  fit_data     = biliverdin_rabsa_data,
  fixed_params = fixed_params_bvd,
  fit_spec     = fit_spec_bvd,
  
  signal_cols = c(
    CD400.7 = "CD400.7",
    CD383.5 = "CD383.5",
    CD410.9 = "CD410.9",
    CD437.6 = "CD437.6"
  ),
  
  channel_scale = "mad",
  
  method           = "nloptr",
  nloptr_algorithm = "NLOPT_LN_BOBYQA",
  loss             = "sse",
  
  nloptr_control = list(
    maxeval     = 10000,
    xtol_rel    = 1e-6,
    print_level = 1
  ),
  
  verbose = TRUE
)

fit_bvd$logB_vec
fit_bvd$response_vec

ggplot(
  fit_bvd$prediction_long,
  aes(x = total_G)
) +
  geom_point(
    aes(y = observed),
    shape = 1
  ) +
  geom_line(
    aes(y = predicted)
  ) +
  facet_wrap(
    ~ channel,
    scales = "free_y"
  ) +
  theme_bw()

# > fit_bvd$logB_vec
# [1]  7.038496  8.412706 20.723036 11.545463
# > fit_bvd$response_vec
# $CD400.7
# [1]    1781.204       0.000 -859322.826 -370844.707 -437416.676  -10198.001
# 
# $CD383.5
# [1]     1037.673        0.000  -558341.344  -809027.718 -1411430.335    11809.938
# 
# $CD410.9
# [1]    9381.092       0.000 -946816.456  506010.527   74822.456   -1699.417
# 
# $CD437.6
# [1]     275.7782       0.0000 -302053.8502   21573.5439  126385.1407 -102386.6803

## Applying BayesianMCMC on competitive models

model_spec_bvd <- tribble(
  ~model,
  ~species_vec,
  ~logB_start,
  ~response_startCD400.7,
  ~response_startCD383.5,
  ~response_startCD410.9,
  ~response_startCD437.6,
  
  "model1",
  c("H1G1", "H1G2"),
  c(5.447015, 10.237214),
  c(2340.07,   0.00,   -669188.21,   -268153.76),
  c(746.0065,  0.0000, -493095.4609, -1297129.2622),
  c(-687.5352, 0.0000, -727780.9883,  336951.7392),
  c(-936.6557, 0.0000, -278530.3147,  296865.0449),
  
  "model2",
  c("H1G1", "H1G2", "H1G3"),
  c(7.5, 8.6, 21.56),
  c(2259.611, 0.000, -574439.818, -379145.905, -468174.889),
  c(1160.079, 0.000, -457259.095, -823786.360, -1353474.045),
  c(9628.404, 0.000, -600416.614,  509389.490,  17902.115),
  c(163.562,  0.000, -218725.744,  4541.109,    148677.189),
  
  "model3",
  c("H1G1", "H1G2", "H1G3", "H2G1"),
  c(7.04, 8.41, 20.72, 11.55),
  c(1.8e3, 0,    -8.6e5, -3.7e5, -4.45,  -1.0e4),
  c(1.0e3, 0,    -5.6e5, -8.1e5, -1.4e6,  1.2e4),
  c(9.4e3, 0,    -9.5e5,  5.1e5,  7.5e4, -1.7e3),
  c(2.7e2, 0,    -3.0e5,  2.2e4,  1.3e5, -1.0e5)
)

make_HG_model_fit_spec <- function(
    species_vec,
    logB_start,
    response_startCD400.7,
    response_startCD383.5,
    response_startCD410.9,
    response_startCD437.6,
    logB_map_scale = 1
) {
  
  n_species <- length(species_vec)
  
  list(
    
    logB_vec = list(
      start     = logB_start,
      lower     = rep(-Inf, n_species),
      upper     = rep( Inf, n_species),
      fit       = rep(TRUE, n_species),
      map_scale = rep(logB_map_scale, n_species)
    ),
    
    response_vec = list(
      
      CD400.7 = list(
        start = response_startCD400.7,
        
        lower = c(
          -Inf, 0,
          rep(-Inf, n_species)
        ),
        
        upper = c(
          Inf, 0,
          rep(Inf, n_species)
        ),
        
        fit = c(
          TRUE, FALSE,
          rep(TRUE, n_species)
        ),
        
        map_scale = c(
          abs(response_startCD400.7[1]), NA,
          abs(response_startCD400.7[-c(1:2)])
        )
      ),
      CD383.5 = list(
        start = response_startCD383.5,
        
        lower = c(
          -Inf, 0,
          rep(-Inf, n_species)
        ),
        
        upper = c(
          Inf, 0,
          rep(Inf, n_species)
        ),
        
        fit = c(
          TRUE, FALSE,
          rep(TRUE, n_species)
        ),
        
        map_scale = c(
          abs(response_startCD383.5[1]), NA,
          abs(response_startCD383.5[-c(1:2)])
        )
      ),
      CD410.9 = list(
        start = response_startCD410.9,
        
        lower = c(
          -Inf, 0,
          rep(-Inf, n_species)
        ),
        
        upper = c(
          Inf, 0,
          rep(Inf, n_species)
        ),
        
        fit = c(
          TRUE, FALSE,
          rep(TRUE, n_species)
        ),
        
        map_scale = c(
          abs(response_startCD410.9[1]), NA,
          abs(response_startCD410.9[-c(1:2)])
        )
      ),
      CD437.6 = list(
        start = response_startCD437.6,
        
        lower = c(
          -Inf, 0,
          rep(-Inf, n_species)
        ),
        
        upper = c(
          Inf, 0,
          rep(Inf, n_species)
        ),
        
        fit = c(
          TRUE, FALSE,
          rep(TRUE, n_species)
        ),
        
        map_scale = c(
          abs(response_startCD437.6[1]), NA,
          abs(response_startCD437.6[-c(1:2)])
        )
      )
      
    )
  )
}

model_spec_bvd <- model_spec_bvd |>
  mutate(
    fit_spec = pmap(
      list(
        species_vec,
        logB_start,
        response_startCD400.7,
        response_startCD383.5,
        response_startCD410.9,
        response_startCD437.6
      ),
      make_HG_model_fit_spec
    )
  )

fit_grid_bvd <- tidyr::crossing(
  model = model_spec_bvd$model
)

fit_grid_bvd <- fit_grid_bvd |>
  mutate(
    fit = map(
      model,
      function(model_name) {
        
        model_row <- model_spec_bvd |>
          filter(model == model_name)
        
        fixed_params <- fixed_params_mel
        
        fixed_params$species_vec <- model_row$species_vec[[1]]
        
        ## ----------------------------------------------------------
        ## Fit
        ## ----------------------------------------------------------
        
        fit_HG_multichannel_nloptr_MCMC(
          fit_data     = biliverdin_rabsa_data,
          fixed_params = fixed_params,
          fit_spec     = model_row$fit_spec[[1]],
          
          signal_cols = c(
            CD400.7 = "CD400.7",
            CD383.5 = "CD383.5",
            CD410.9 = "CD410.9",
            CD437.6 = "CD437.6"
          ),
          
          fitter_args = list(
            channel_scale = "mad",
            
            method           = "nloptr",
            nloptr_algorithm = "NLOPT_LN_BOBYQA",
            loss             = "sse",
            
            nloptr_control = list(
              maxeval     = 10000,
              xtol_rel    = 1e-6,
              print_level = 1
            ),
            
            verbose = TRUE
          ),
          
          mcmc_args = list(
            iterations = 4000L,
            burnin     = 1000L,
            n_internal_chains = 4L,
            n_Z = 100L,
            seed = 1
          )
        )
      }
    )
  )

dir.create(
  "HG_fits_biliverdin_rabsa",
  showWarnings = FALSE
)

pwalk(
  fit_grid_bvd,
  function(
    model,
    fit
  ) {
    
    saveRDS(
      fit,
      file = file.path(
        "HG_fits_biliverdin_rabsa",
        paste0(
          model,
          ".rds"
        )
      )
    )
  }
)

fits_bvd <- setNames(
  fit_grid_bvd$fit,
  fit_grid_bvd$model
)

fit_summary_bvd <- fit_grid_bvd |>
  mutate(
    objective = map_dbl(
      fit,
      ~ .x$fit$objective_value
    ),
    
    npar = map_int(
      fit,
      ~ length(.x$fit$par)
    ),
    
    convergence = map_int(
      fit,
      ~ .x$fit$fit$status
    )
  ) |>
  select(
    model,
    npar,
    objective,
    convergence
  )

fit_summary_bvd

fit_parameters_bvd <- fit_grid_bvd |>
  transmute(
    model,
    
    parameter_table = map(
      fit,
      function(x) {
        
        det <- x$fit
        
        det$par_table |>
          mutate(
            estimate = unname(
              det$par[name]
            )
          ) |>
          select(
            name,
            group,
            channel,
            species,
            estimate,
            start,
            lower,
            upper
          )
      }
    )
  ) |>
  tidyr::unnest(
    parameter_table
  )

fit_parameters_bvd

posterior_summary_bvd <- fit_grid_bvd |>
  transmute(
    model,
    
    posterior_summary = map(
      fit,
      ~ .x$posterior_summary
    )
  ) |>
  tidyr::unnest(
    posterior_summary
  )

posterior_summary_bvd

posterior_logB_bvd <- posterior_summary_bvd |>
  filter(
    grepl("^logB_", parameter)
  )

posterior_logB_bvd

posterior_response_bvd <- posterior_summary_bvd |>
  filter(
    grepl("^response_", parameter)
  )

fit_predictions_bvd <- fit_grid_bvd |>
  transmute(
    model,
    
    prediction = map(
      fit,
      ~ .x$fit$prediction_long
    )
  ) |>
  tidyr::unnest(
    prediction
  )

fit_predictions_bvd

channel_fit_metrics_bvd <- fit_predictions_bvd |>
  group_by(
    model,
    channel
  ) |>
  summarise(
    n = n(),
    
    SSE = sum(residual^2),
    
    RMSE = sqrt(mean(residual^2)),
    
    MAE = mean(abs(residual)),
    
    .groups = "drop"
  )

channel_fit_metrics_bvd

model_comparison_bvd <- fit_grid_bvd |>
  mutate(
    metrics = map(
      fit,
      extract_HG_MC_model_metrics
    )
  ) |>
  select(
    model,
    metrics
  ) |>
  tidyr::unnest(
    metrics
  ) |>
  mutate(
    delta_AICc_fixed = AICc_fixed - min(AICc_fixed),
    
    delta_AICc_profile = AICc_profile - min(AICc_profile),
    
    weight_AICc_fixed = {
      w <- exp(-0.5 * delta_AICc_fixed)
      
      w / sum(w)
    },
    
    weight_AICc_profile = {
      w <- exp(-0.5 * delta_AICc_profile)
      
      w / sum(w)
    }
  ) |> 
  arrange(
    AICc_profile
  )

model_comparison_bvd |>
  select(
    model,
    N,
    p,
    SSE,
    AICc_fixed,
    delta_AICc_fixed,
    weight_AICc_fixed,
    AICc_profile,
    delta_AICc_profile,
    weight_AICc_profile
  )

channel_labels <- c(
  "CD383.5" = "383.5 nm",
  "CD400.7" = "400.7 nm",
  "CD410.9" = "410.9 nm",
  "CD437.6" = "437.6 nm"
)

model_labels <- c(
  "model1" = "1:2 binding",
  "model2" = "1:2 and 1:3\nbinding",
  "model3" = "1:2, 2:1, and 1:3\nbinding"
)

plot_fit_bvd <- ggplot(
  fit_predictions_bvd,
  aes(x = total_G, color = channel, linetype = model)
) +
  geom_point(
    aes(y = observed),
    shape = 1,
    size  = 1.5,
    show.legend = FALSE
  ) +
  geom_line(
    aes(y = predicted),
    linewidth = 0.6
  ) +
  facet_grid(
    channel ~ model,
    scales   = "free_y",
    labeller = labeller(channel = channel_labels, model = model_labels)
  ) +
  scale_color_manual(
    values = c(
      "CD383.5" = "#004488",
      "CD400.7" = "#EE7733",
      "CD410.9" = "#00C19F",
      "CD437.6" = "#EE3377"
    ),
    guide  = "none"
  ) +
  scale_linetype_manual(
    values = c(
      "model1" = "solid",
      "model2" = "dashed",
      "model3" = "dotted"
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

plot_fit_bvd

aic_labels_bvd <- fit_predictions_bvd |>
  group_by(
    model,
    channel
  ) |>
  summarise(
    x_min = min(total_G, na.rm = TRUE),
    
    x_max = max(total_G, na.rm = TRUE),
    
    .groups = "drop"
  ) |>
  mutate(
    x = (x_min + x_max) / 2
  ) |>
  left_join(
    model_comparison_bvd |>
      select(
        model,
        delta_AICc_profile
      ),
    by = "model"
  ) |>
  mutate(
    label = sprintf(
      "\u0394AICc = %.2f",
      delta_AICc_profile
    )
  )

sigma_bvd <- fit_grid_bvd |>
  transmute(
    model,
    sigma = map(
      fit,
      function(x) {
        
        x$posterior_summary |>
          filter(
            grepl("^sigma_", parameter)
          ) |>
          transmute(
            channel_safe = sub(
              "^sigma_",
              "",
              parameter
            ),
            sigma_mean  = mean,
            sigma_lower = lower,
            sigma_upper = upper
          )
      }
    )
  ) |>
  unnest(sigma) |>
  mutate(
    channel = case_when(
      channel_safe == "CD383_5" ~ "CD383.5",
      channel_safe == "CD400_7" ~ "CD400.7",
      channel_safe == "CD410_9" ~ "CD410.9",
      channel_safe == "CD437_6" ~ "CD437.6",
      TRUE                      ~ channel_safe
    )
  )

plot_residuals_bvd <- ggplot(
  fit_predictions_bvd,
  aes(x = total_G, y = residual, color = channel, linetype = model)
) +
  geom_rect(
    data = sigma_bvd,
    aes(
      xmin = -Inf,
      xmax = Inf,
      ymin = -2 * sigma_mean,
      ymax =  2 * sigma_mean
    ),
    inherit.aes = FALSE,
    alpha       = 0.08
  ) +
  geom_hline(
    yintercept = 0,
    linewidth  = 0.35
  ) +
  geom_point(
    shape = 1,
    size  = 1.5
  ) +
  geom_line(
    linewidth = 0.6
  ) +
  geom_text(
    data = aic_labels_bvd,
    aes(x = x, y = Inf, label = label),
    inherit.aes = FALSE,
    hjust = 0.5,
    vjust = 1.3,
    size  = 3
  ) +
  facet_grid(
    channel ~ model,
    scales   = "free_y",
    labeller = labeller(channel = channel_labels, model = model_labels)
  ) +
  scale_color_manual(
    values = c(
      "CD383.5" = "#004488",
      "CD400.7" = "#EE7733",
      "CD410.9" = "#00C19F",
      "CD437.6" = "#EE3377"
    ),
    guide  = "none"
  ) +
  scale_linetype_manual(
    values = c(
      "model1" = "solid",
      "model2" = "dashed",
      "model3" = "dotted"
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
    y = "Residual"
  ) +
  theme_bw() +
  theme(
    strip.background = element_blank()
  )

plot_residuals_bvd

plot_bvd <- grid.arrange(
  plot_fit_bvd,
  plot_residuals_bvd,
  nrow = 1
)

ggsave(
  "plot_bvd.svg",
  plot_bvd,
  width  = 15,
  height = 8
)

################################################################################

# Trying multistart search to improve CD383.5 fit

dir.create(
  "HG_fits_biliverdin_multistart",
  showWarnings = FALSE
)

signal_cols_bvd <- c(
  CD400.7 = "CD400.7",
  CD383.5 = "CD383.5",
  CD410.9 = "CD410.9",
  CD437.6 = "CD437.6"
)

candidate_models_bvd <- list(
  
  H1G1_H1G2 =
    c("H1G1", "H1G2"),
  
  H1G1a_H1G1b_H1G2 =
    c("H1G1a", "H1G1b", "H1G2"),
  
  H1G1_H1G3 =
    c("H1G1", "H1G3"),
  
  H1G1_H1G2_H1G3 =
    c("H1G1", "H1G2", "H1G3")
)

## ---------------------------------------------------------------------------
## Sensible random cumulative-logB range from stoichiometric order
##
## At concentrations around 10^-5 M:
##
## H1G1  -> log beta ~ 4-8
## H1G2  -> log beta ~ 7-14
## H1G3  -> log beta ~ 10-20
##
## These are deliberately broad.
## ---------------------------------------------------------------------------

make_logB_bounds <- function(species_vec) {
  
  sp <- parse_species(species_vec)
  
  molecularity_minus_1 <- sp$nH + sp$nG - 1
  
  center <- 4.5 * molecularity_minus_1
  
  tibble(
    lower = pmax(
      center - 5,
      -2
    ),
    
    upper = pmin(
      center + 6,
      24
    )
  )
}

## ---------------------------------------------------------------------------
## Random multichannel fit specification
## ---------------------------------------------------------------------------

make_random_bvd_fit_spec <- function(
    species_vec,
    data,
    signal_cols
) {
  
  n_species <- length(species_vec)
  
  ## ---- log beta ------------------------------------------------------------
  
  K_bounds <- make_logB_bounds(species_vec)
  
  logB_start <- runif(
    n_species,
    min = K_bounds$lower,
    max = K_bounds$upper
  )
  
  ## ---- channel-specific response starts -----------------------------------
  
  response_specs <- map(
    signal_cols,
    function(signal_name) {
      
      y <- data[[signal_name]]
      
      ## Characteristic response coefficient:
      ##
      ## signal ~ response * concentration
      response_scale <- max(abs(y), na.rm = TRUE) / median(data$total_H, na.rm = TRUE)
      
      response_scale <- max(response_scale, 1e4)
      
      ## Free host response from G = 0 point.
      i0 <- which.min(data$total_G)
      
      response_H_start <- y[i0] /data$total_H[i0]
      
      ## Random complex responses.
      ##
      ## Both signs deliberately allowed because CD signals can
      ## reverse sign between species/wavelengths.
      complex_start <- runif(
        n_species,
        min = -1.5 * response_scale,
        max =  1.5 * response_scale
      )
      
      start <- c(
        response_H_start,
        0,
        complex_start
      )
      
      bound_response <- 5 * response_scale
      
      list(
        start = start,
        
        lower = c(
          -bound_response,
          0,
          rep(
            -bound_response,
            n_species
          )
        ),
        
        upper = c(
          bound_response,
          0,
          rep(
            bound_response,
            n_species
          )
        ),
        
        fit = c(
          TRUE,
          FALSE,
          rep(
            TRUE,
            n_species
          )
        ),
        
        map_scale = c(
          max(
            abs(response_H_start),
            response_scale / 10
          ),
          
          NA,
          
          pmax(
            abs(complex_start),
            response_scale / 10
          )
        )
      )
    }
  )
  
  names(response_specs) <- names(signal_cols)
  
  
  list(
    
    logB_vec = list(
      start = logB_start,
      lower = K_bounds$lower,
      upper = K_bounds$upper,
      fit   = rep(
        TRUE,
        n_species
      ),
      map_scale = rep(
        2,
        n_species
      )
    ),
    
    response_vec = response_specs
  )
}

run_bvd_multistart <- function(
    n_start = 100,
    seed    = 1
) {
  
  set.seed(seed)
  
  results <- list()
  
  run_id <- 0L
  
  for (
    model_name in names(candidate_models_bvd)
  ) {
    
    species_vec <- candidate_models_bvd[[model_name]]
    
    fixed_params <- fixed_params_bvd
    
    fixed_params$species_vec <- species_vec
    
    
    for (
      start_id in seq_len(n_start)
    ) {
      
      run_id <- run_id + 1L
      
      cat(
        "\n====================================================\n",
        model_name,
        " | start ",
        start_id,
        "/",
        n_start,
        "\n",
        sep = ""
      )
      
      fit_spec <- make_random_bvd_fit_spec(
          species_vec = species_vec,
          data        = biliverdin_rabsa_data,
          signal_cols = signal_cols_bvd
        )
      
      fit <- tryCatch(
        
        fit_HG_multichannel_titration(
          fit_data         = biliverdin_rabsa_data,
          fixed_params     = fixed_params,
          fit_spec         = fit_spec,
          signal_cols      = signal_cols_bvd,
          channel_scale    = "none",
          method           = "nloptr",
          nloptr_algorithm = "NLOPT_LN_BOBYQA",
          loss             = "sse",
          
          nloptr_control = list(
            maxeval     = 5000,
            xtol_rel    = 1e-6,
            print_level = 0
          ),
          
          verbose = FALSE
        ),
        
        error = function(e) {
          message(
            "FAILED: ",
            conditionMessage(e)
          )
          
          NULL
        }
      )
      
      if (is.null(fit)) {
        next
      }
      
      ## -----------------------------------------------------------
      ## Save complete fit immediately
      ## -----------------------------------------------------------
      
      filename <- file.path(
        "HG_fits_biliverdin_multistart",
        sprintf(
          "%s_%03d.rds",
          model_name,
          start_id
        )
      )
      
      saveRDS(
        fit,
        filename
      )
      
      ## -----------------------------------------------------------
      ## Lightweight leaderboard information
      ## -----------------------------------------------------------
      
      results[[length(results) + 1L]] <- tibble(
        
        model       = model_name,
        start_id    = start_id,
        objective   = fit$objective_value,
        convergence = fit$fit$status,
        logB        = list(fit$logB_vec),
        par         = list(fit$par),
        file        = filename
      )
      
      leaderboard <- bind_rows(
        results
      ) |>
        arrange(
          objective
        )
      
      saveRDS(
        leaderboard,
        file.path(
          "HG_fits_biliverdin_multistart",
          "leaderboard.rds"
        )
      )
      
      write.csv(
        leaderboard |>
          select(
            model,
            start_id,
            objective,
            convergence,
            file
          ),
        
        file.path(
          "HG_fits_biliverdin_multistart",
          "leaderboard.csv"
        ),
        
        row.names = FALSE
      )
      
      cat(
        "objective = ",
        signif(
          fit$objective_value,
          7
        ),
        "\n"
      )
    }
  }
  
  bind_rows(
    results
  ) |>
    arrange(
      objective
    )
}

multi_bvd <- run_bvd_multistart(
  n_start = 50,
  seed    = 123
)

# inspect
multi_bvd |>
  select(
    model,
    start_id,
    objective,
    convergence,
    logB
  ) |>
  print(
    n = 30
  )

best_bvd <- multi_bvd |>
  group_by(
    model
  ) |>
  slice_min(
    objective,
    n         = 1,
    with_ties = FALSE
  ) |>
  ungroup()

best_bvd

best_fit_bvd <- readRDS(
  multi_bvd$file[
    which.min(
      multi_bvd$objective
    )
  ]
)

best_fit_bvd$logB_vec
best_fit_bvd$response_vec
best_fit_bvd$prediction_long

ggplot(
  best_fit_bvd$prediction_long,
  aes(
    x = total_G
  )
) +
  geom_point(
    aes(
      y = observed
    ),
    shape = 1
  ) +
  geom_line(
    aes(
      y = predicted
    )
  ) +
  facet_wrap(
    ~ channel,
    scales = "free_y"
  ) +
  theme_bw()
################################################################################
## Simulation examples of Host-Guest titration fit
################################################################################

## Loading environment ---------------------------------------------------------

source("01_HGtitr_simulator.R")
source("02_HGtitr_fitter.R")
source("03_HGtitr_uncertainty_estimator.R")
library(plotly)
library(processx)
library(dplyr)
library(purrr)
library(tibble)
library(tidyr)
library(ggplot2)
library(scales)
library(signs)

set.seed(0)

## Demonstration of thin-plate spline interpolation ----------------------------

HG_surface_to_matrices <- function(
    sim,
    grid_only = TRUE
) {
  
  dat <- sim$surface
  
  if (
    grid_only &&
    "point_type" %in% names(dat)
  ) {
    dat <- dat[
      dat$point_type == "grid",
      ,
      drop = FALSE
    ]
  }
  
  required <- c(
    "free_H",
    "free_G",
    "total_H",
    "total_G",
    "signal"
  )
  
  if (!all(required %in% names(dat))) {
    stop(
      "sim$surface must contain: ",
      paste(required, collapse = ", ")
    )
  }
  
  dat <- dat[
    is.finite(dat$free_H)  &
      is.finite(dat$free_G)  &
      is.finite(dat$total_H) &
      is.finite(dat$total_G) &
      is.finite(dat$signal),
    ,
    drop = FALSE
  ]
  
  ## --------------------------------------------------------------
  ## Recover original free-concentration mesh coordinates
  ## --------------------------------------------------------------
  
  H_axis <- sort(unique(dat$free_H))
  G_axis <- sort(unique(dat$free_G))
  
  iH <- match(
    dat$free_H,
    H_axis
  )
  
  iG <- match(
    dat$free_G,
    G_axis
  )
  
  nH <- length(H_axis)
  nG <- length(G_axis)
  
  ## --------------------------------------------------------------
  ## Create same-sized matrices
  ##
  ## rows    = free_H
  ## columns = free_G
  ## --------------------------------------------------------------
  
  total_H_mat <- matrix(
    NA_real_,
    nrow = nH,
    ncol = nG
  )
  
  total_G_mat <- matrix(
    NA_real_,
    nrow = nH,
    ncol = nG
  )
  
  signal_mat <- matrix(
    NA_real_,
    nrow = nH,
    ncol = nG
  )
  
  total_H_mat[cbind(iH, iG)] <- dat$total_H
  total_G_mat[cbind(iH, iG)] <- dat$total_G
  signal_mat[cbind(iH, iG)]  <- dat$signal
  
  list(
    total_H = total_H_mat,
    total_G = total_G_mat,
    signal  = signal_mat,
    
    free_H_axis = H_axis,
    free_G_axis = G_axis,
    
    dimensions = c(
      nH = nH,
      nG = nG
    ),
    
    retained_cells    = nrow(dat),
    total_cells       = nH * nG,
    retained_fraction = nrow(dat) / (nH * nG)
  )
}

camera_3d <- function(
    azimuth    = 45,
    elevation  = 30,
    distance   = 2,
    projection = "perspective"
) {
  
  az <- azimuth   * pi / 180
  el <- elevation * pi / 180
  
  list(
    eye = list(
      x = distance * cos(el) * cos(az),
      y = distance * cos(el) * sin(az),
      z = distance * sin(el)
    ),
    center = list(x = 0, y = 0, z = 0),
    up     = list(x = 0, y = 0, z = 1),
    projection = list(
      type = "perspective"
    )
  )
}

sim_tps <- simulate_HG_titration(
  species_vec  = c("H1G1", "H1G2"),
  logB_vec     = c(5, 8),
  response_vec = c(0, 0, 2e5, 3e5),
  nsim         = c(100, 500),
  c_stock_H    = 1e-6,
  c_stock_G    = 1e-3,
  V_stock_H    = 3.00,
  V_G          = seq(0, 0.1, length.out = 100),
)

surf_mat <- HG_surface_to_matrices(sim_tps$sim)

plot_surf <- plot_ly(
  x = surf_mat$total_G,
  y = surf_mat$total_H,
  z = surf_mat$signal,
  type        = "surface",
  connectgaps = FALSE,
  
  contours = list(
    x = list(
      show  = TRUE,
      color = "black",
      width = 1,
      start = 0,
      end   = 6e-4,
      size  = 1e-4
    ),
    y = list(
      show  = TRUE,
      color = "black",
      width = 1,
      start = 0,
      end   = 1e-6,
      size  = 2e-7
    )
  )
) |>
  layout(
    margin = list(
      t = 0,
      b = 0,
      l = 0,
      r = 0
    ),
    
    scene = list(
      camera = camera_3d(
        azimuth    = -130,
        elevation  = 20,
        distance   = 2.2,
        projection = "orthographic"
      ),
      
      xaxis = list(
        title = "Total [G]"
      ),
      yaxis = list(
        title = "Total [H]",
        range = c(4.0e-7, 9.5e-7)
      ),
      zaxis = list(
        title = "Signal"
      )
    )
  )

orca(
  plot_surf,
  "plot_surf.svg"
)

plot_totalG <- ggplot(data = sim1$tps_result$prediction) +
  geom_point(
    mapping = aes(x = total_G, y = signal_pred)
  ) +
  labs(
    title    = "Response signal vs total [Guest]",
    subtitle = "at near-fixed total [Host]",
    x        = expression("[G]"[total]),
    y        = "signal"
  ) +
  theme_bw() +
  theme(
    plot.title    = element_text(hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    axis.title    = element_text(size = 12),
    axis.text     = element_text(size = 12)
  )

plot_freeG <- ggplot(data = sim1$sim$surface |> filter(total_H > 9.95e-7 & total_H < 1e-6)) +
  geom_point(
    mapping = aes(x = -log10(free_G), y = signal),
    shape   = 1,
    color   = "red"
  ) +
  geom_function(
    fun   = function(x) (0 + 0.2 * 10^(5 - x) + 0.3 * 10^(8 - 2 * x)) / (1 + 10^(5 - x) + 10^(8 - 2 * x)),
    color = "red"
  ) +
  labs(
    title    = "Response signal vs free [Guest]",
    subtitle = "and theoretical curve",
    x        = expression(-log[10]~"[G]"[free])
  ) +
  theme_bw() +
  theme(
    plot.title    = element_text(hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    axis.title    = element_text(size = 12),
    axis.text     = element_text(size = 12)
  )

ggsave(
  filename = "plot_totalG.svg",
  plot     =  plot_totalG,
  height   = 5,
  width    = 5
)
ggsave(
  filename = "plot_freeG.svg",
  plot     =  plot_freeG,
  height   = 5,
  width    = 5
)

## Fit comparison of 4 examples ------------------------------------------------

## Creating simulated data
make_HG_scenario <- function(
    species_vec,
    logB_vec,
    response_vec,
    common_params,
    noise_sd                  = 1e-3,
    seed                      = NULL,
    logB_map_scale            = 2,
    response_map_scale_factor = 1
) {
  
  ## --------------------------------------------------------------
  ## Checks
  ## --------------------------------------------------------------
  
  stopifnot(
    is.character(species_vec),
    is.numeric(logB_vec),
    is.numeric(response_vec),
    length(logB_vec) == length(species_vec),
    length(response_vec) == length(species_vec) + 2L
  )
  
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  ## --------------------------------------------------------------
  ## 1. Generate exact simulated titration
  ## --------------------------------------------------------------
  
  sim_args <- c(
    list(
      species_vec  = species_vec,
      logB_vec     = logB_vec,
      response_vec = response_vec
    ),
    common_params
  )
  
  sim <- do.call(
    simulate_HG_titration,
    sim_args
  )
  
  ## --------------------------------------------------------------
  ## 2. Generate noisy experimental data
  ## --------------------------------------------------------------
  
  fit_data <- sim$tps_result$prediction |>
    select(
      total_H,
      total_G,
      signal_pred
    ) |>
    mutate(
      signal = signal_pred + rnorm(n(), mean = 0, sd = noise_sd)
    )
  
  ## --------------------------------------------------------------
  ## 3. Fixed parameters
  ##
  ## Everything needed by simulate_HG_titration except the fitted
  ## logB_vec and response_vec.
  ## --------------------------------------------------------------
  
  fixed_params <- c(
    list(
      species_vec = species_vec
    ),
    common_params[
      setdiff(
        names(common_params),
        c(
          "logB_vec",
          "response_vec"
        )
      )
    ]
  )
  
  ## --------------------------------------------------------------
  ## 4. Fitting specification
  ## --------------------------------------------------------------
  
  n_species <- length(species_vec)
  
  ## Characteristic response scales.
  response_scale <- c(
    NA_real_,
    NA_real_,
    pmax(
      abs(response_vec[-c(1, 2)]) * response_map_scale_factor,
      1
    )
  )
  
  fit_spec <- list(
    
    logB_vec = list(
      start = logB_vec,
      lower = rep(-Inf, n_species),
      upper = rep( Inf, n_species),
      fit   = rep(TRUE, n_species),
      map_scale = rep(
        logB_map_scale,
        n_species
      )
    ),
    
    response_vec = list(
      start = response_vec,
      lower = c(
        response_vec[1:2],
        rep(-Inf, n_species)
      ),
      upper = c(
        response_vec[1:2],
        rep( Inf, n_species)
      ),
      fit = c(
        FALSE,
        FALSE,
        rep(TRUE, n_species)
      ),
      map_scale = response_scale
    )
  )
  
  list(
    sim          = sim,
    fit_data     = fit_data,
    fixed_params = fixed_params,
    fit_spec     = fit_spec,
    
    ## Useful metadata
    truth = list(
      species_vec  = species_vec,
      logB_vec     = logB_vec,
      response_vec = response_vec,
      noise_sd     = noise_sd
    )
  )
}

common_params <- list(
  
  nsim = c(100, 500),
  
  c_stock_H = 1e-6,
  c_stock_G = 1e-1,
  V_stock_H = 3,
  
  V_G = seq(0, 0.01, length.out = 30),
  
  max_train = 300
)

scenario_spec <- tribble(
  ~scenario,    ~species_vec,        ~logB_vec,   ~response_vec,
  
  "scenario1",   c("H1G1"),           c(4),        c(0, 0, 3e5),
  
  "scenario2",   c("H1G1", "H1G2"),   c(4,   2),   c(0, 0, 3e5, 4e5),
  
  "scenario3",   c("H1G1", "H1G2"),   c(4.5, 8),   c(0, 0, 2e5, 3e5),
  
  "scenario4",   c("H1G1", "H1G2"),   c(5,   8),   c(0, 0, 2e5, 3e5)
)

scenarios <- pmap(
  scenario_spec[
    c(
      "species_vec",
      "logB_vec",
      "response_vec"
    )
  ],
  \(species_vec, logB_vec, response_vec) {
    
    make_HG_scenario(
      species_vec   = species_vec,
      logB_vec      = logB_vec,
      response_vec  = response_vec,
      common_params = common_params,
      noise_sd      = 1e-3
    )
  }
)

names(scenarios) <- scenario_spec$scenario

## Fitting different models

## ------------------------------------------------------------------
## Candidate models
## ------------------------------------------------------------------

model_spec <- tribble(
  ~model,    ~species_vec,               ~logB_start,  ~response_start,
  
  "model1",   c("H1G1"),                  c(4),         c(0, 0, 3e5),
  
  "model2",   c("H1G1", "H1G2"),          c(5, 8),      c(0, 0, 2e5, 3e5),
  
  "model3",   c("H1G1", "H1G2"),          c(4, 2),      c(0, 0, 3e5, 4e5),
  
  "model4",   c("H1G1", "H1G2", "H1G3"),  c(5, 6, 8),   c(0, 0, 2e5, 2.5e5, 3e5)
)

make_HG_model_fit_spec <- function(
    species_vec,
    logB_start,
    response_start,
    logB_map_scale = 2
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
      start = response_start,
      
      lower = c(
        response_start[1:2],
        rep(-Inf, n_species)
      ),
      
      upper = c(
        response_start[1:2],
        rep(Inf, n_species)
      ),
      
      fit = c(
        FALSE,
        FALSE,
        rep(TRUE, n_species)
      ),
      
      map_scale = c(
        NA_real_,
        NA_real_,
        pmax(
          abs(response_start[-c(1, 2)]),
          1
        )
      )
    )
  )
}

model_spec <- model_spec |>
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

fit_grid <- tidyr::crossing(
  scenario = names(scenarios),
  model    = model_spec$model
)

fit_grid <- fit_grid |>
  mutate(
    fit = map2(
      scenario,
      model,
      function(scenario_name, model_name) {
        
        scenario_obj <- scenarios[[scenario_name]]
        
        model_row <- model_spec |>
          filter(model == model_name)
        
        ## ----------------------------------------------------------
        ## Fixed parameters come from the experimental scenario,
        ## BUT species_vec comes from the candidate fitted model.
        ## ----------------------------------------------------------
        
        fixed_params <- scenario_obj$fixed_params
        
        fixed_params$species_vec <- model_row$species_vec[[1]]
        
        ## ----------------------------------------------------------
        ## Fit
        ## ----------------------------------------------------------
        
        fit_HG_titration(
          fit_data = scenario_obj$fit_data,
          
          fixed_params = fixed_params,
          
          fit_spec = model_row$fit_spec[[1]],
          
          signal_col = "signal",
          
          method = "nloptr",
          nloptr_algorithm = "NLOPT_LN_BOBYQA",
          
          loss = "sse",
          
          nloptr_control = list(
            maxeval     = 300,
            xtol_rel    = 1e-5,
            print_level = 0
          ),
          
          verbose = TRUE
        )
      }
    )
  )

fit_summary <- fit_grid |>
  mutate(
    objective = map_dbl(
      fit,
      "objective_value"
    ),
    
    npar = map_int(
      fit,
      ~ length(.x$par)
    ),
    
    convergence = map_int(
      fit,
      ~ .x$fit$status
    )
  ) |>
  select(
    scenario,
    model,
    npar,
    objective,
    convergence
  )

fit_parameters <- fit_grid |>
  transmute(
    scenario,
    model,
    parameter = map(
      fit,
      ~ tibble(
        parameter = names(.x$par),
        estimate  = as.numeric(.x$par)
      )
    )
  ) |>
  unnest(parameter)

dir.create(
  "HG_fits",
  showWarnings = FALSE
)

pwalk(
  fit_grid,
  function(
    scenario,
    model,
    fit
  ) {
    
    saveRDS(
      fit,
      file = file.path(
        "HG_fits",
        paste0(
          scenario,
          "_",
          model,
          ".rds"
        )
      )
    )
  }
)

# Loading back works
# fit_grid <- tidyr::crossing(
#   scenario = paste0("scenario", 1:4),
#   model    = paste0("model", 1:4)
# ) |>
#   mutate(
#     fit = purrr::map2(
#       scenario,
#       model,
#       ~ readRDS(
#         file.path(
#           "HG_fits",
#           paste0(.x, "_", .y, ".rds")
#         )
#       )
#     )
#   )

## Compare models

model_comparison <- fit_grid |>
  mutate(
    n = map_int(
      fit,
      ~ nrow(.x$prediction)
    ),
    
    p = map_int(
      fit,
      ~ length(.x$par)
    ),
    
    SSE = map_dbl(
      fit,
      ~ sum(
        .x$prediction$residual^2,
        na.rm = TRUE
      )
    ),
    
    RMSE = map_dbl(
      fit,
      ~ sqrt(
        mean(
          .x$prediction$residual^2,
          na.rm = TRUE
        )
      )
    ),
    
    ## Include sigma as an estimated likelihood parameter.
    k = p + 1L,
    
    logLik = map2_dbl(
      fit,
      n,
      ~ {
        sse <- sum(
          .x$prediction$residual^2,
          na.rm = TRUE
        )
        
        -.y / 2 * (log(2 * pi) + 1 + log(sse / .y))
      }
    ),
    
    AIC = -2 * logLik + 2 * k,
    
    AICc = AIC + 2 * k * (k + 1) / (n - k - 1)
  ) |>
  group_by(scenario) |>
  mutate(
    delta_AIC  = AIC - min(AIC),
    delta_AICc = AICc - min(AICc),
    
    AIC_weight = {
      w <- exp(-0.5 * delta_AIC)
      w / sum(w)
    },
    
    AICc_weight = {
      w <- exp(-0.5 * delta_AICc)
      w / sum(w)
    }
  ) |>
  ungroup() |>
  arrange(
    scenario,
    AICc
  ) |>
  select(
    scenario,
    model,
    n,
    p,
    SSE,
    RMSE,
    AIC,
    AICc,
    delta_AIC,
    delta_AICc,
    AIC_weight,
    AICc_weight
  )

model_comparison |>
  select(
    scenario,
    model,
    p,
    SSE,
    AICc,
    delta_AICc,
    AICc_weight
  )

## Plotting diagnostics

fit_predictions <- fit_grid |>
  transmute(
    scenario,
    model,
    prediction = map(
      fit,
      ~ .x$prediction |>
        select(
          total_H,
          total_G,
          signal,
          signal_pred,
          residual
        )
    )
  ) |>
  tidyr::unnest(prediction)

# Custom formatting function
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

plot_fits <- ggplot(data    = fit_predictions,
                    mapping = aes(x = total_G, color = model)) +
  geom_point(
    aes(y = signal),
    size  = 1.5,
    shape = 1
  ) +
  geom_line(
    aes(y = signal_pred),
    linewidth = 0.7
  ) +
  scale_x_continuous(
    breaks = c(0, 1e-3, 2e-3, 3e-3, 4e-3),
    labels = label_scientific
  ) +
  facet_grid(
    scenario ~ model,
    scales = "free_y"
  ) +
  labs(
    x = expression("[G]"[total]),
    y = "signal"
  ) +
  theme_bw() +
  theme(
    legend.position  = "none",
    axis.text.x      = element_text(angle = 45, vjust = 0.5),
    strip.background = element_blank()
  )

ggsave(
  filename = "plot_fits.svg",
  plot     =  plot_fits,
  height   = 6,
  width    = 6
)

plot_fits_logscale <- plot_fits +
  scale_x_continuous(
    breaks = c(0, 1e-4, 1e-3),
    labels = label_scientific,
    trans  = scales::pseudo_log_trans(
      sigma = 1e-5,
      base  = 10
    )
  ) +
  labs(
    x = expression("[G]"[total]~" (pseudo-logarithmic scale)"),
  )

ggsave(
  filename = "plot_fits_logscale.svg",
  plot     =  plot_fits_logscale,
  height   = 6,
  width    = 6
)

pseudo_trans <- scales::pseudo_log_trans(
  sigma = 1e-5,
  base  = 10
)

delta_labels <- fit_predictions |>
  group_by(scenario, model) |>
  summarise(
    x_min = min(total_G, na.rm = TRUE),
    x_max = max(total_G, na.rm = TRUE),
    .groups = "drop"
  ) |>
  mutate(
    x = pseudo_trans$inverse((pseudo_trans$transform(x_min) + pseudo_trans$transform(x_max)) / 2)
  ) |>
  left_join(
    model_comparison |>
      select(
        scenario,
        model,
        delta_AICc
      ),
    by = c("scenario", "model")
  ) |>
  mutate(
    label = sprintf(
      "\u0394AICc = %.1f",
      delta_AICc
    )
  )

plot_residuals <- ggplot(data    = fit_predictions,
                         mapping = aes(x = total_G, y = residual, color = model)) +
  geom_hline(
    yintercept = c(-2e-3, 2e-3),
    linetype   = 3
  ) +
  geom_point(
    size  = 1.5,
    shape = 1
  ) +
  geom_line(
    linewidth = 0.5
  ) +
  geom_text(
    data        = delta_labels,
    mapping     = aes(x = x, y = Inf, label = label),
    inherit.aes = FALSE,
    vjust       = 1.4,
    hjust       = 0.5,
    size        = 3
  ) +
  scale_x_continuous(
    breaks = c(0, 1e-4, 1e-3),
    labels = label_scientific,
    trans  = scales::pseudo_log_trans(
      sigma = 1e-5,
      base  = 10
    )
  ) +
  scale_y_continuous(labels = signs_format()) +
  facet_grid(
    scenario ~ model,
    scales   = "fixed"
  ) +
  labs(
    x = expression("[G]"[total]~" (pseudo-logarithmic scale)"),
    y = "residual"
  ) +
  theme_bw() +
  theme(
    legend.position  = "none",
    axis.text.x      = element_text(angle = 45, vjust = 0.5),
    strip.background = element_blank()
  )

ggsave(
  filename = "plot_residuals.svg",
  plot     =  plot_residuals,
  height   = 6,
  width    = 6
)

## Job plot analysis with 4 examples -------------------------------------------

make_Job_design <- function(
    C_total,
    n            = 101L,
    X_G          = NULL,
    endpoint_eps = 1e-6
) {
  
  stopifnot(
    length(C_total) == 1L,
    is.finite(C_total),
    C_total > 0
  )
  
  if (is.null(X_G)) {
    ## Avoid exact total_H = 0 because the current TPS machinery
    ## uses log(total_H). Exact X_G = 0 is fine because total_G
    ## already has an offset in the interpolator.
    X_G <- seq(0, 1 - endpoint_eps, length.out = n)
  }
  
  if (
    any(!is.finite(X_G)) ||
    any(X_G < 0) ||
    any(X_G > 1)
  ) {
    stop("`X_G` must lie between 0 and 1.")
  }
  
  data.frame(
    X_G     = X_G,
    total_H = C_total * (1 - X_G),
    total_G = C_total * X_G
  )
}

simulate_Job_scenario <- function(
    scenario,
    C_total,
    n            = 151L,
    endpoint_eps = 1e-6
) {
  
  if (is.null(scenario$truth)) {
    stop("Scenario does not contain `$truth`.")
  }
  
  if (is.null(scenario$fixed_params)) {
    stop("Scenario does not contain `$fixed_params`.")
  }
  
  design <- make_Job_design(
    C_total      = C_total,
    n            = n,
    endpoint_eps = endpoint_eps
  )
  
  fixed <- scenario$fixed_params
  
  ## Remove the ordinary titration design.
  fixed[c(
    "species_vec",
    "logB_vec",
    "response_vec",
    "c_stock_H",
    "c_stock_G",
    "V_stock_H",
    "V_G",
    "c_HG_customdf"
  )] <- NULL
  
  sim_args <- fixed
  sim_args$species_vec   <- scenario$truth$species_vec
  sim_args$logB_vec      <- scenario$truth$logB_vec
  sim_args$response_vec  <- scenario$truth$response_vec
  sim_args$c_HG_customdf <- design[, c("total_H", "total_G"), drop = FALSE]
  
  sim <- do.call(
    simulate_HG_titration,
    sim_args
  )
  
  pred <- sim$tps_result$prediction
  
  ## Recalculate X_G from the actual prediction coordinates rather
  ## than relying on row ordering.
  pred <- pred |>
    dplyr::mutate(
      total_conc = total_H + total_G,
      X_G        = total_G / total_conc,
      
      ## Signal expected in the absence of complex formation,
      ## using the free H and G response coefficients.
      baseline_signal =
        scenario$truth$response_vec[1] * total_H +
        scenario$truth$response_vec[2] * total_G,
      
      ## Classical Job-like excess response.
      delta_signal = signal_pred - baseline_signal
    ) |>
    dplyr::arrange(X_G)
  
  list(
    design     = design,
    sim        = sim,
    prediction = pred,
    C_total    = C_total
  )
}

job_concentrations <- c(1e-6, 1e-5, 1e-4, 1e-3, 1e-2)

sim_grid <- tidyr::expand_grid(
  scenario_name = names(scenarios),
  C_total       = job_concentrations
)

fused_job_data <- purrr::pmap_dfr(
  sim_grid,
  function(scenario_name, C_total) {
    
    current_scenario <- scenarios[[scenario_name]]
    
    sim_result <- simulate_Job_scenario(
      scenario = current_scenario,
      C_total  = C_total,
      n        = 151
    )
    
    sim_result$prediction |>
      dplyr::mutate(
        scenario = scenario_name,
        C_total  = C_total
      )
  }
)

fused_job_data <- fused_job_data |> 
  group_by(scenario, C_total) |> 
  mutate(
    delta_signal_norm = delta_signal / max(delta_signal)
  )

max_points <- fused_job_data |> 
  dplyr::group_by(scenario, C_total) |> 
  dplyr::slice_max(delta_signal, n = 1, with_ties = FALSE) |> 
  dplyr::ungroup()

plot_Job <- ggplot(
  fused_job_data,
  aes(
    x      = X_G,
    y      = delta_signal_norm,
    group  = factor(C_total),
    colour = factor(C_total)
  )
) +
  geom_line(linewidth = 0.8) +
  geom_vline(
    data = max_points,
    aes(xintercept = X_G, colour = factor(C_total)),
    linetype = "dashed",
    alpha    = 0.7
  ) +
  facet_wrap(~ scenario, scales = "free_y") + 
  labs(
    x      = expression(italic(X)[G] == frac("[G]"[total], "[H]"[total] + "[G]"[total])),
    y      = expression(Delta * "Signal (max normalized)"),
    title  = "Job plots",
    colour = expression(C[total])
  ) +
  theme_bw() +
  theme(
    strip.background = element_rect(fill = "gray95"),
    strip.text       = element_text(face = "bold"),
    plot.title       = element_text(hjust = 0.5)
  )

ggsave(
  filename = "plot_Job2.svg",
  plot     =  plot_Job,
  height   = 6,
  width    = 7
)

## Uncertainty analysis shown with 2-step separate example ---------------------

fit_bayes_scenario4 <- fit_HG_nloptr_MCMC(
  fit_data                = scenarios$scenario4$fit_data,
  fixed_params            = scenarios$scenario4$fixed_params,
  fit_spec                = scenarios$scenario4$fit_spec,
  signal_col              = "signal",
  logB_half_width         = 2,
  response_rel_half_width = 2,
  sigma_bounds_factor     = c(0.1, 10),
  iterations              = 4000,
  burnin                  = 1000,
  n_internal_chains       = 4,
  n_Z                     = 100,
  seed                    = 0
)

fit_bayes_scenario4$fit$par
fit_bayes_scenario4$posterior_summary

BayesianTools::tracePlot(
  fit_bayes_scenario4$bayes,
  parametersOnly = TRUE
)

BayesianTools::marginalPlot(
  fit_bayes_scenario4$bayes
)

BayesianTools::correlationPlot(
  fit_bayes_scenario4$bayes,
  parametersOnly = TRUE
)

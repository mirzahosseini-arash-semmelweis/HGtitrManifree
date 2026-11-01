################################################################################
## Uncertainty estimation for multichannel Host-Guest titration fitting
##
## Shared logB_vec + channel-specific response_vec + channel-specific sigma.
##
## Prerequisites:
##   source("01_HGtitr_simulator.R")
##   source("02_HGtitr_fitter.R")
##   source("04_HGtitr_multichannel_fitter.R")
##
## Bayesian likelihood:
##
##   y_ic ~ Normal(
##     signal_ic(logB_vec, response_vec[[c]]),
##     sigma_c^2
##   )
##
## with the same equilibrium constants in all channels and one independently
## estimated residual standard deviation per channel.
##
## The default likelihood assumes conditional independence of channels.
################################################################################


## Residual-scale initialization -----------------------------------------------

estimate_sigma_from_HG_multichannel_fit <- function(
    fit,
    method = c("rmse_df", "rmse", "mad")
) {
  method <- match.arg(method)

  if (
    is.null(fit$prediction_long) ||
    is.null(fit$par_table)       ||
    is.null(fit$channels)
  ) {
    stop(
      "`fit` does not contain the required multichannel fit components."
    )
  }

  channels <- fit$channels
  n_shared <- sum(fit$par_table$group == "logB_vec")
  sigma        <- numeric(length(channels))
  names(sigma) <- channels

  for (ch in channels) {
    res <- fit$prediction_long$residual[fit$prediction_long$channel == ch]
    res <- as.numeric(res)
    res <- res[is.finite(res)]

    if (length(res) < 2L) {
      stop(
        sprintf(
          "Too few finite residuals in channel `%s`.",
          ch
        )
      )
    }

    n_response <- sum(
      fit$par_table$group == "response_vec" &
      fit$par_table$channel == ch
    )

    ## Allocate the shared-logB degrees of freedom equally across channels.
    p_effective <- n_response + n_shared / length(channels)

    sigma[ch] <- switch(
      method,

      rmse_df = {
        df <- max(
          length(res) - p_effective,
          1
        )

        sqrt(sum(res^2) / df)
      },

      rmse = sqrt(mean(res^2)),

      mad = stats::mad(
        res,
        center   = stats::median(res),
        constant = 1.4826
      )
    )
  }

  if (
    any(!is.finite(sigma)) ||
    any(sigma <= 0)
  ) {
    stop(
      "Could not obtain positive sigma estimates for every channel."
    )
  }

  sigma
}


## Finite Bayesian parameter domain --------------------------------------------

make_HG_multichannel_bayes_fit_spec <- function(
    fit,
    logB_half_width         = 2,
    response_rel_half_width = 2,
    response_map_half_width = 1,
    clip_to_original_bounds = TRUE
) {
  if (
    is.null(fit$fit_spec)  ||
    is.null(fit$par_table) ||
    is.null(fit$par)
  ) {
    stop(
      "`fit` must contain `fit_spec`, `par_table`, and `par`."
    )
  }

  spec <- fit$fit_spec

  par_tbl <- as.data.frame(
    fit$par_table,
    stringsAsFactors = FALSE
  )

  best_par <- unlist(fit$par)

  for (i in seq_len(nrow(par_tbl))) {
    nm      <- par_tbl$name[i]
    group   <- par_tbl$group[i]
    index   <- par_tbl$index[i]
    channel <- par_tbl$channel[i]

    center <- unname(best_par[nm])

    if (!is.finite(center)) {
      stop(
        "Non-finite fitted parameter: ",
        nm
      )
    }

    if (group == "logB_vec") {

      half_width <- logB_half_width

      group_spec <- spec$logB_vec

    } else if (group == "response_vec") {

      map_scale <- par_tbl$map_scale[i]

      candidate_widths <- c(
        response_rel_half_width * abs(center),

        if (is.finite(map_scale)) {
          response_map_half_width * map_scale
        } else {
          NA_real_
        }
      )

      candidate_widths <- candidate_widths[
        is.finite(candidate_widths) &
        candidate_widths > 0
      ]

      if (!length(candidate_widths)) {
        half_width <- max(
          abs(center),
          1
        )
      } else {
        half_width <- max(
          candidate_widths
        )
      }

      group_spec <- spec$response_vec[[channel]]

    } else {
      stop(
        "Unknown parameter group: ",
        group
      )
    }

    new_lower <- center - half_width
    new_upper <- center + half_width
    old_lower <- group_spec$lower[index]
    old_upper <- group_spec$upper[index]

    if (isTRUE(clip_to_original_bounds)) {

      if (is.finite(old_lower)) {
        new_lower <- max(
          new_lower,
          old_lower
        )
      }

      if (is.finite(old_upper)) {
        new_upper <- min(
          new_upper,
          old_upper
        )
      }
    }

    if (
      new_lower >= center ||
      new_upper <= center
    ) {
      warning(
        sprintf(
          paste0(
            "Deterministic estimate `%s` lies at or extremely near ",
            "the proposed Bayesian boundary."
          ),
          nm
        )
      )
    }

    if (group == "logB_vec") {

      spec$logB_vec$start[index] <- center
      spec$logB_vec$lower[index] <- new_lower
      spec$logB_vec$upper[index] <- new_upper

    } else {

      spec$response_vec[[channel]]$start[index] <- center
      spec$response_vec[[channel]]$lower[index] <- new_lower
      spec$response_vec[[channel]]$upper[index] <- new_upper
    }
  }

  spec
}

## Bayesian predictor ----------------------------------------------------------

make_HG_multichannel_bayes_predictor <- function(
    fit,
    bayes_fit_spec
) {
  if (
    !exists(
      "prepare_HG_multichannel_fit_spec",
      mode = "function"
    )
  ) {
    stop(
      "`prepare_HG_multichannel_fit_spec()` is not available. ",
      "Source 04_HGtitr_multichannel_fitter.R first."
    )
  }

  if (
    is.null(fit$fixed_params$species_vec) ||
    is.null(fit$data_used)                ||
    is.null(fit$channels)
  ) {
    stop(
      "`fit` is missing fixed parameters, data, or channel metadata."
    )
  }

  channels <- fit$channels

  prep <- prepare_HG_multichannel_fit_spec(
    fit_spec    = bayes_fit_spec,
    species_vec = fit$fixed_params$species_vec,
    channels    = channels
  )

  par_tbl  <- prep$par_table
  fit_spec <- prep$fit_spec

  theta0        <- par_tbl$start
  names(theta0) <- par_tbl$name

  lower        <- par_tbl$lower
  names(lower) <- par_tbl$name

  upper        <- par_tbl$upper
  names(upper) <- par_tbl$name

  if (
    any(!is.finite(lower)) ||
    any(!is.finite(upper))
  ) {
    stop(
      paste0(
        "Finite lower and upper bounds are required because the ",
        "Bayesian implementation uses bounded uniform priors."
      )
    )
  }

  parameter_names <- par_tbl$name
  total_H <- as.numeric(fit$data_used$total_H)
  total_G <- as.numeric(fit$data_used$total_G)

  endpoint_tol <- if (
    is.null(fit$fixed_params$endpoint_tol)
  ) {
    1e-7
  } else {
    fit$fixed_params$endpoint_tol
  }

  endpoint_maxiter <- if (
    is.null(fit$fixed_params$endpoint_maxiter)
  ) {
    500L
  } else {
    fit$fixed_params$endpoint_maxiter
  }

  endpoint_damping <- if (
    is.null(fit$fixed_params$endpoint_damping)
  ) {
    0.5
  } else {
    fit$fixed_params$endpoint_damping
  }

  use_optim_fallback <- if (
    is.null(fit$fixed_params$use_optim_fallback)
  ) {
    TRUE
  } else {
    fit$fixed_params$use_optim_fallback
  }

  predict_raw <- function(
      theta = theta0
  ) {
    theta <- unlist(theta)

    if (is.null(names(theta))) {

      if (
        length(theta) !=
        length(parameter_names)
      ) {
        stop(
          "Unnamed theta has incorrect length."
        )
      }

      names(theta) <- parameter_names
    }

    missing_parameters <- setdiff(
      parameter_names,
      names(theta)
    )

    if (length(missing_parameters)) {
      stop(
        "theta is missing: ",
        paste(
          missing_parameters,
          collapse = ", "
        )
      )
    }

    theta <- theta[parameter_names]

    if (any(!is.finite(theta))) {
      stop(
        "theta contains non-finite values."
      )
    }

    fitted_vectors <- par_vector_to_HG_multichannel_vectors(
        par      = theta,
        par_tbl  = par_tbl,
        fit_spec = fit_spec
      )

    pred_obj <- predict_HG_multichannel_at_totals(
        total_H            = total_H,
        total_G            = total_G,
        species_vec        = fit$fixed_params$species_vec,
        logB_vec           = fitted_vectors$logB_vec,
        response_vec       = fitted_vectors$response_vec,
        endpoint_tol       = endpoint_tol,
        endpoint_maxiter   = endpoint_maxiter,
        endpoint_damping   = endpoint_damping,
        use_optim_fallback = use_optim_fallback
      )

    predicted <- pred_obj$pred

    if (
      !identical(
        colnames(predicted),
        channels
      )
    ) {
      predicted <- predicted[
        ,
        channels,
        drop = FALSE
      ]
    }

    if (any(!is.finite(predicted))) {
      stop(
        "Prediction returned non-finite values."
      )
    }

    predicted
  }

  structure(
    list(
      predict_raw     = predict_raw,
      theta0          = theta0,
      lower           = lower,
      upper           = upper,
      par_table       = par_tbl,
      fit_spec        = fit_spec,
      parameter_names = parameter_names,
      channels        = channels,
      safe_channels   = prep$safe_channels,
      fixed_params    = fit$fixed_params,
      total_H         = total_H,
      total_G         = total_G
    ),
    class = "HG_multichannel_fit_predictor"
  )
}

## Bayesian model --------------------------------------------------------------

make_HG_multichannel_bayes_model <- function(
    fit,
    bayes_fit_spec,
    sigma_start         = NULL,
    sigma_bounds_factor = c(
      0.1,
      10
    )
) {
  if (!requireNamespace("BayesianTools", quietly = TRUE)) {
    stop("Package `BayesianTools` is required.")
  }

  predictor <- make_HG_multichannel_bayes_predictor(
    fit            = fit,
    bayes_fit_spec = bayes_fit_spec
  )

  channels <- predictor$channels

  observed <- fit$observed_matrix[
    ,
    channels,
    drop = FALSE
  ]

  if (any(!is.finite(observed))) {
    stop(
      "Observed multichannel data contain non-finite values."
    )
  }

  test_prediction <- predictor$predict_raw()

  if (!identical(dim(test_prediction), dim(observed))) {
    stop(
      "Observed and predicted multichannel matrices have different dimensions."
    )
  }

  if (is.null(sigma_start)) {
    sigma_start <-
      estimate_sigma_from_HG_multichannel_fit(
        fit,
        method = "rmse_df"
      )
  }

  if (length(sigma_start) == 1L) {
    sigma_start <- rep(
      sigma_start,
      length(channels)
    )
  }

  if (!is.null(names(sigma_start))) {

    if (!all(channels %in% names(sigma_start))) {
      stop(
        "`sigma_start` is missing channel(s): ",
        paste(
          setdiff(channels, names(sigma_start)),
          collapse = ", "
        )
      )
    }

    sigma_start <- sigma_start[channels]
  }

  sigma_start <- as.numeric(sigma_start)

  names(sigma_start) <- channels

  if (
    length(sigma_start) != length(channels) ||
    any(!is.finite(sigma_start))            ||
    any(sigma_start <= 0)
  ) {
    stop(
      "`sigma_start` must contain one positive finite value per channel."
    )
  }

  if (
    length(sigma_bounds_factor) != 2L    ||
    any(!is.finite(sigma_bounds_factor)) ||
    any(sigma_bounds_factor <= 0)        ||
    sigma_bounds_factor[1L] >=
      sigma_bounds_factor[2L]
  ) {
    stop(
      "`sigma_bounds_factor` must contain two positive increasing values."
    )
  }

  sigma_lower <- sigma_start * sigma_bounds_factor[1L]
  sigma_upper <- sigma_start * sigma_bounds_factor[2L]
  safe_channels <- predictor$safe_channels[channels]

  sigma_parameter_names <- paste0(
    "log_sigma_",
    safe_channels
  )

  theta0 <- c(
    predictor$theta0,
    stats::setNames(
      log(sigma_start),
      sigma_parameter_names
    )
  )

  lower <- c(
    predictor$lower,
    stats::setNames(
      log(sigma_lower),
      sigma_parameter_names
    )
  )

  upper <- c(
    predictor$upper,
    stats::setNames(
      log(sigma_upper),
      sigma_parameter_names
    )
  )

  parameter_names <- names(theta0)

  log_likelihood <- function(
      par
  ) {
    par <- unlist(par)

    if (is.null(names(par))) {

      if (
        length(par) !=
        length(parameter_names)
      ) {
        return(-Inf)
      }

      names(par) <- parameter_names
    }

    if (
      any(
        !parameter_names %in%
        names(par)
      )
    ) {
      return(-Inf)
    }

    par <- par[parameter_names]

    if (any(!is.finite(par))) {
      return(-Inf)
    }

    theta <- par[predictor$parameter_names]

    sigma <- exp(par[sigma_parameter_names])

    if (
      any(!is.finite(sigma)) ||
      any(sigma <= 0)
    ) {
      return(-Inf)
    }

    predicted <- tryCatch(
      predictor$predict_raw(
        theta
      ),
      error = function(e) {
        NULL
      }
    )

    if (
      is.null(predicted) ||
      !identical(
        dim(predicted),
        dim(observed)
      )                  ||
      any(!is.finite(predicted))
    ) {
      return(-Inf)
    }

    ll <- 0

    for (j in seq_along(channels)) {
      ll <- ll +
        sum(
          stats::dnorm(
            observed[, j],
            mean = predicted[, j],
            sd   = sigma[j],
            log  = TRUE
          )
        )
    }

    if (!is.finite(ll)) {
      -Inf
    } else {
      ll
    }
  }

  prior <- BayesianTools::createUniformPrior(
    lower = lower,
    upper = upper,
    best  = theta0
  )

  setup <- BayesianTools::createBayesianSetup(
    likelihood = log_likelihood,
    prior      = prior,
    names      = parameter_names
  )

  structure(
    c(
      predictor,
      list(
        observed              = observed,
        sigma_start           = sigma_start,
        sigma_bounds          = rbind(
          lower = sigma_lower,
          upper = sigma_upper
        ),
        sigma_parameter_names = sigma_parameter_names,
        bayes_theta0          = theta0,
        bayes_lower           = lower,
        bayes_upper           = upper,
        bayes_parameter_names = parameter_names,
        log_likelihood        = log_likelihood,
        prior                 = prior,
        setup                 = setup
      )
    ),
    class = c(
      "HG_multichannel_bayes_model",
      class(predictor)
    )
  )
}

## DEzs population helpers -----------------------------------------------------

make_HG_multichannel_bounded_cloud <- function(
    center,
    lower,
    upper,
    n,
    z_sd = 0.10
) {
  center <- unlist(center)

  parameter_names <- names(center)

  lower <- lower[parameter_names]

  upper <- upper[parameter_names]

  if (
    any(!is.finite(center)) ||
    any(!is.finite(lower))  ||
    any(!is.finite(upper))
  ) {
    stop(
      "center, lower, and upper must all be finite."
    )
  }

  if (any(lower >= upper)) {
    stop(
      "All lower bounds must be below upper bounds."
    )
  }

  if (length(z_sd) == 1L) {
    z_sd <- rep(
      z_sd,
      length(center)
    )
  }

  if (length(z_sd) != length(center)) {
    stop(
      "`z_sd` must be scalar or one value per parameter."
    )
  }

  result <- matrix(
    NA_real_,
    nrow = n,
    ncol = length(center),
    dimnames = list(
      NULL,
      parameter_names
    )
  )

  for (j in seq_along(center)) {

    relative_position <- (center[j] - lower[j]) / (upper[j] - lower[j])

    relative_position <- pmin(
      pmax(
        relative_position,
        1e-8
      ),
      1 - 1e-8
    )

    center_logit <- stats::qlogis(relative_position)

    sampled_logit <- center_logit +
      stats::rnorm(
        n,
        mean = 0,
        sd   = z_sd[j]
      )

    result[, j] <- lower[j] + (upper[j] - lower[j]) *
      stats::plogis(
        sampled_logit
      )
  }

  result
}

make_HG_multichannel_bayes_populations <- function(
    bayes_model,
    n_internal_chains = 8L,
    n_Z               = 150L,
    start_sd          = 0.08,
    Z_sd              = 0.12,
    seed              = 0
) {
  if (!is.null(seed)) {
    set.seed(seed)
  }

  start_value <- make_HG_multichannel_bounded_cloud(
      center = bayes_model$bayes_theta0,
      lower  = bayes_model$bayes_lower,
      upper  = bayes_model$bayes_upper,
      n      = n_internal_chains,
      z_sd   = start_sd
    )

  Z <- make_HG_multichannel_bounded_cloud(
    center = bayes_model$bayes_theta0,
    lower  = bayes_model$bayes_lower,
    upper  = bayes_model$bayes_upper,
    n      = n_Z,
    z_sd   = Z_sd
  )

  ## Include the exact deterministic optimum.
  start_value[1L, ] <- bayes_model$bayes_theta0

  Z[1L, ] <- bayes_model$bayes_theta0

  list(
    startValue = start_value,
    Z          = Z
  )
}

## Posterior summary -----------------------------------------------------------

posterior_summary_HG_multichannel <- function(
    posterior,
    conf_level = 0.95
) {
  alpha <- 1 - conf_level

  data.frame(
    parameter = names(
      posterior
    ),

    mean = vapply(
      posterior,
      mean,
      numeric(1)
    ),

    median = vapply(
      posterior,
      median,
      numeric(1)
    ),

    sd = vapply(
      posterior,
      stats::sd,
      numeric(1)
    ),

    lower = vapply(
      posterior,
      stats::quantile,
      numeric(1),
      probs = alpha / 2,
      names = FALSE
    ),

    upper = vapply(
      posterior,
      stats::quantile,
      numeric(1),
      probs = 1 - alpha / 2,
      names = FALSE
    ),

    row.names = NULL
  )
}

## Bayesian uncertainty from an existing deterministic fit ---------------------

fit_HG_multichannel_MCMC <- function(
    fit,

    ## ---- Automatic finite Bayesian bounds ----------------------
    bayes_fit_spec = NULL,

    logB_half_width         = 2,
    response_rel_half_width = 2,
    response_map_half_width = 1,

    clip_to_original_bounds = TRUE,

    ## ---- Channel-specific sigma -------------------------------
    sigma_method = c(
      "rmse_df",
      "rmse",
      "mad"
    ),

    sigma_start = NULL,

    sigma_bounds_factor = c(
      0.1,
      10
    ),

    ## ---- DEzs -------------------------------------------------
    iterations = 12000L,
    burnin     = 3000L,
    thin       = 1L,

    n_internal_chains = 8L,
    n_Z               = 150L,

    start_sd = 0.08,
    Z_sd     = 0.12,

    seed = 0,

    consoleUpdates = 1000L,

    mcmc_extra_settings = list(),

    verbose = TRUE
) {
  if (!requireNamespace("BayesianTools", quietly = TRUE)) {
    stop(
      "Package `BayesianTools` is required."
    )
  }

  if (!inherits(fit, "HG_multichannel_titration_fit")) {
    stop(
      "`fit` must be an HG_multichannel_titration_fit object."
    )
  }

  sigma_method <- match.arg(sigma_method)

  if (is.null(bayes_fit_spec)) {

    bayes_fit_spec <- make_HG_multichannel_bayes_fit_spec(
        fit                     = fit,
        logB_half_width         = logB_half_width,
        response_rel_half_width = response_rel_half_width,
        response_map_half_width = response_map_half_width,
        clip_to_original_bounds = clip_to_original_bounds
      )
  }

  if (is.null(sigma_start)) {
    sigma_start <- estimate_sigma_from_HG_multichannel_fit(
        fit,
        method = sigma_method
      )
  }

  bayes_model <- make_HG_multichannel_bayes_model(
      fit                 = fit,
      bayes_fit_spec      = bayes_fit_spec,
      sigma_start         = sigma_start,
      sigma_bounds_factor = sigma_bounds_factor
    )

  populations <- make_HG_multichannel_bayes_populations(
      bayes_model       = bayes_model,
      n_internal_chains = n_internal_chains,
      n_Z               = n_Z,
      start_sd          = start_sd,
      Z_sd              = Z_sd,
      seed              = seed
    )

  mcmc_settings <- utils::modifyList(

    list(
      iterations     = iterations,
      burnin         = burnin,
      thin           = thin,
      startValue     = populations$startValue,
      Z              = populations$Z,
      f              = 1.0,
      pGamma1        = 0.01,
      pSnooker       = 0.01,
      eps.mult       = 0.05,
      consoleUpdates = consoleUpdates,
      message        = verbose
    ),

    mcmc_extra_settings
  )

  bayes <- BayesianTools::runMCMC(
    bayesianSetup = bayes_model$setup,
    sampler       = "DEzs",
    settings      = mcmc_settings
  )

  posterior <- as.data.frame(
    BayesianTools::getSample(
      bayes,
      parametersOnly = TRUE
    )
  )

  physical_sigma_names <- paste0(
    "sigma_",
    bayes_model$safe_channels[
      bayes_model$channels
    ]
  )

  for (j in seq_along(bayes_model$channels)) {
    posterior[physical_sigma_names[j]] <- exp(
      posterior[bayes_model$sigma_parameter_names[j]]
    )
  }

  posterior_parameters <- posterior[
    ,
    c(
      bayes_model$parameter_names,
      physical_sigma_names
    ),
    drop = FALSE
  ]

  posterior_stats <- posterior_summary_HG_multichannel(
      posterior_parameters
    )

  out <- list(
    fit               = fit,
    bayes_fit_spec    = bayes_fit_spec,
    sigma_start       = sigma_start,
    bayes_model       = bayes_model,
    populations       = populations,
    bayes             = bayes,
    posterior         = posterior,
    posterior_summary = posterior_stats
  )

  class(out) <- "HG_multichannel_MCMC_fit"

  out
}

## Optional all-in-one deterministic + Bayesian wrapper ------------------------

fit_HG_multichannel_nloptr_MCMC <- function(
    fit_data,
    fixed_params,
    fit_spec,
    signal_cols,
    fitter_args = list(),
    mcmc_args   = list()
) {
  deterministic_args <- utils::modifyList(
    list(
      fit_data     = fit_data,
      fixed_params = fixed_params,
      fit_spec     = fit_spec,
      signal_cols  = signal_cols
    ),
    fitter_args
  )

  deterministic_fit <- do.call(
    fit_HG_multichannel_titration,
    deterministic_args
  )

  uncertainty_args <- utils::modifyList(
    list(
      fit = deterministic_fit
    ),
    mcmc_args
  )

  do.call(
    fit_HG_multichannel_MCMC,
    uncertainty_args
  )
}

## Helper for extracting model metrics -----------------------------------------

extract_HG_MC_model_metrics <- function(
    mcmc_fit
) {
  
  ## Deterministic fit nested inside MCMC object
  det <- mcmc_fit$fit
  
  pred <- det$prediction_long
  
  channel_stats <- pred |>
    group_by(channel) |>
    summarise(
      n = n(),
      
      SSE = sum(residual^2),
      
      RMSE = sqrt(mean(residual^2)),
      
      .groups = "drop"
    )
  
  ## ------------------------------------------------------------
  ## Numbers of observations and fitted chemical parameters
  ## ------------------------------------------------------------
  
  N <- sum(channel_stats$n)
  
  p <- length(det$par)
  
  n_channels <- nrow(channel_stats)
  
  ## ------------------------------------------------------------
  ## A. Fixed-scale likelihood
  ##
  ## Exactly corresponds to deterministic optimization because
  ## channel_scale was fixed during fitting.
  ## ------------------------------------------------------------
  
  scale_vec <- det$channel_scale[channel_stats$channel]
  
  logLik_fixed <- sum(
    -0.5 * channel_stats$n *
      log(2 * pi * scale_vec^2) -
      channel_stats$SSE / (2 * scale_vec^2)
  )
  
  k_fixed <- p
  
  AIC_fixed <- -2 * logLik_fixed + 2 * k_fixed
  
  AICc_fixed <- AIC_fixed + 2 * k_fixed * (k_fixed + 1) / (N - k_fixed - 1)
  
  ## ------------------------------------------------------------
  ## B. Profile likelihood:
  ## separate unknown sigma for each channel
  ## ------------------------------------------------------------
  
  SSE_safe <- pmax(
    channel_stats$SSE,
    .Machine$double.xmin
  )
  
  logLik_profile <- sum(
    -channel_stats$n / 2 * (log(2 * pi) + 1 + log(SSE_safe / channel_stats$n))
  )
  
  ## fitted chemical parameters + one sigma per channel
  k_profile <- p + n_channels
  
  AIC_profile <- -2 * logLik_profile + 2 * k_profile
  
  AICc_profile <- AIC_profile + 2 * k_profile * (k_profile + 1) / (N - k_profile - 1)
  
  tibble(
    N = N,
    p = p,
    n_channels = n_channels,
    
    SSE = sum(channel_stats$SSE),
    
    RMSE = sqrt(sum(channel_stats$SSE) / N),
    
    logLik_fixed = logLik_fixed,
    AIC_fixed    = AIC_fixed,
    AICc_fixed   = AICc_fixed,
    
    logLik_profile = logLik_profile,
    AIC_profile    = AIC_profile,
    AICc_profile   = AICc_profile
  )
}

## Example ---------------------------------------------------------------------

if (FALSE) { ### turn TRUE to run example
  ## --------------------------------------------------------------------------
  ## First run the example in 04_HGtitr_multichannel_fitter.R, which creates:
  ##
  ##   fit_data_mc
  ##   fixed_params_mc
  ##   fit_spec_mc
  ##   fit_mc
  ## --------------------------------------------------------------------------

  source("01_HGtitr_simulator.R")
  source("02_HGtitr_fitter.R")
  source("04_HGtitr_multichannel_fitter.R")
  source("05_HGtitr_multichannel_uncertainty_estimator.R")

  ## If fit_mc does not already exist, first run:
  ##
  ## fit_mc <- fit_HG_multichannel_titration(...)
  ##
  ## from the example in 04_HGtitr_multichannel_fitter.R.

  estimate_sigma_from_HG_multichannel_fit(
    fit_mc,
    method = "rmse_df"
  )

  ## --------------------------------------------------------------------------
  ## Pilot MCMC
  ## --------------------------------------------------------------------------
  fit_mc_bayes <- fit_HG_multichannel_MCMC(
    fit = fit_mc,

    logB_half_width         = 2,
    response_rel_half_width = 2,
    response_map_half_width = 1,

    sigma_bounds_factor = c(
      0.1,
      10
    ),

    iterations = 4000L,
    burnin     = 1000L,

    n_internal_chains = 4L,
    n_Z               = 100L,

    seed = 1,

    consoleUpdates = 500L,
    verbose        = TRUE
  )

  fit_mc_bayes$fit$logB_vec
  fit_mc_bayes$fit$response_vec
  fit_mc_bayes$sigma_start
  fit_mc_bayes$posterior_summary

  BayesianTools::tracePlot(
    fit_mc_bayes$bayes,
    parametersOnly = TRUE
  )

  BayesianTools::marginalPlot(
    fit_mc_bayes$bayes
  )

  BayesianTools::correlationPlot(
    fit_mc_bayes$bayes,
    parametersOnly = TRUE
  )

  ## --------------------------------------------------------------------------
  ## All-in-one equivalent
  ## --------------------------------------------------------------------------
  fit_mc_bayes2 <- fit_HG_multichannel_nloptr_MCMC(
    fit_data     = fit_data_mc,
    fixed_params = fixed_params_mc,
    fit_spec     = fit_spec_mc,

    signal_cols = c(
      CD = "signal_CD",
      FL = "signal_FL"
    ),

    fitter_args = list(
      channel_scale = c(
        CD = 1e-3,
        FL = 2e-3
      ),

      method = "nloptr",

      nloptr_algorithm =
        "NLOPT_LN_BOBYQA",

      loss = "sse",

      nloptr_control = list(
        maxeval     = 250,
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
  
  fit_mc_bayes2$fit$logB_vec
  fit_mc_bayes2$fit$response_vec
  fit_mc_bayes2$sigma_start
  fit_mc_bayes2$posterior_summary
}

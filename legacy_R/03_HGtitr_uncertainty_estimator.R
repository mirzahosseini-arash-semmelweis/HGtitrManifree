################################################################################
## Uncertainty estimation of Host-Guest titration fit
################################################################################

## Loading libraries -----------------------------------------------------------

library(BayesianTools)

## Bayesian estimation of uncertainty ------------------------------------------

estimate_sigma_from_HG_fit <- function(
    fit,
    method = c("rmse_df", "rmse", "mad")
) {
  
  method <- match.arg(method)
  
  if (
    is.null(fit$prediction) ||
    is.null(fit$prediction$residual)
  ) {
    stop("`fit$prediction$residual` is missing.")
  }
  
  res <- as.numeric(fit$prediction$residual)
  
  res <- res[is.finite(res)]
  
  if (length(res) < 2L) {
    stop("Too few finite residuals.")
  }
  
  n <- length(res)
  p <- length(fit$par)
  
  sigma <- switch(
    method,
    
    rmse_df = {
      df <- max(n - p, 1L)
      sqrt(sum(res^2) / df)
    },
    
    rmse = {
      sqrt(mean(res^2))
    },
    
    mad = {
      stats::mad(
        res,
        center = median(res),
        constant = 1.4826
      )
    }
  )
  
  if (
    !is.finite(sigma) ||
    sigma <= 0
  ) {
    stop("Could not obtain a positive sigma estimate.")
  }
  
  sigma
}

make_HG_bayes_fit_spec <- function(
    fit,
    logB_half_width         = 2,
    response_rel_half_width = 2,
    response_map_half_width = 1,
    clip_to_original_bounds = TRUE
) {
  
  if (is.null(fit$fit_spec)) {
    stop("`fit$fit_spec` is missing.")
  }
  
  if (is.null(fit$par_table)) {
    stop("`fit$par_table` is missing.")
  }
  
  if (is.null(fit$par)) {
    stop("`fit$par` is missing.")
  }
  
  spec <- fit$fit_spec
  
  par_tbl <- as.data.frame(
    fit$par_table,
    stringsAsFactors = FALSE
  )
  
  best_par <- unlist(
    fit$par
  )
  
  for (i in seq_len(nrow(par_tbl))) {
    
    nm    <- par_tbl$name[i]
    group <- par_tbl$group[i]
    index <- par_tbl$index[i]
    
    center <- unname(
      best_par[nm]
    )
    
    if (!is.finite(center)) {
      stop(
        "Non-finite fitted parameter: ",
        nm
      )
    }
    
    ## ------------------------------------------------------------
    ## Choose new prior half-width
    ## ------------------------------------------------------------
    
    if (group == "logB_vec") {
      
      half_width <- logB_half_width
      
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
      
    } else {
      
      stop(
        "Unknown parameter group: ",
        group
      )
    }
    
    new_lower <- center - half_width
    new_upper <- center + half_width
    
    ## ------------------------------------------------------------
    ## Respect original finite physical constraints
    ## ------------------------------------------------------------
    
    old_lower <- spec[[group]]$lower[index]
    old_upper <- spec[[group]]$upper[index]
    
    if (isTRUE(clip_to_original_bounds)) {
      
      if (is.finite(old_lower)) {
        new_lower <- max(new_lower, old_lower)
      }
      
      if (is.finite(old_upper)) {
        new_upper <- min(new_upper, old_upper)
      }
    }
    
    if (new_lower >= center || new_upper <= center) {
      warning(
        sprintf(
          paste0(
            "BOBYQA estimate `%s` lies at or extremely near ",
            "the proposed Bayesian boundary."
          ),
          nm
        )
      )
    }
    
    spec[[group]]$start[index] <- center
    spec[[group]]$lower[index] <- new_lower
    spec[[group]]$upper[index] <- new_upper
  }
  
  spec
}

make_HG_fit_predictor <- function(
    fit,
    bayes_fit_spec
) {
  
  if (is.null(fit$fixed_params)) {
    stop("`fit$fixed_params` is missing.")
  }
  
  if (is.null(fit$fixed_params$species_vec)) {
    stop("`fit$fixed_params$species_vec` is missing.")
  }
  
  prep <- prepare_HG_fit_spec(
    fit_spec    = bayes_fit_spec,
    species_vec = fit$fixed_params$species_vec
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
        "Finite lower and upper bounds are required here because ",
        "the Bayesian model uses a bounded uniform prior."
      )
    )
  }
  
  parameter_names <- par_tbl$name
  
  predict_raw <- function(
    theta = theta0
  ) {
    
    theta <- unlist(theta)
    
    if (is.null(names(theta))) {
      
      if (length(theta) != length(parameter_names)) {
        stop("Unnamed theta has incorrect length.")
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
    
    theta <- theta[
      parameter_names
    ]
    
    if (any(!is.finite(theta))) {
      stop("theta contains non-finite values.")
    }
    
    fitted_vectors <- par_vector_to_HG_vectors(
      par      = theta,
      par_tbl  = par_tbl,
      fit_spec = fit_spec
    )
    
    sim_args <- utils::modifyList(
      fit$fixed_params,
      fitted_vectors
    )
    
    sim <- do.call(
      simulate_HG_titration,
      sim_args
    )
    
    if (
      is.null(sim$tps_result)            ||
      is.null(sim$tps_result$prediction) ||
      is.null(sim$tps_result$prediction$signal_pred)
    ) {
      stop("Simulation produced no signal prediction.")
    }
    
    predicted <- as.numeric(
      sim$tps_result$prediction$signal_pred
    )
    
    if (any(!is.finite(predicted))) {
      stop("Simulation returned non-finite predictions.")
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
      fixed_params    = fit$fixed_params
    ),
    class = "HG_fit_predictor"
  )
}

make_HG_bayes_model <- function(
    fit,
    observed,
    bayes_fit_spec,
    sigma_start         = NULL,
    sigma_bounds_factor = c(0.1, 10)
) {
  
  if (
    !requireNamespace(
      "BayesianTools",
      quietly = TRUE
    )
  ) {
    stop("Package `BayesianTools` is required.")
  }
  
  observed <- as.numeric(
    observed
  )
  
  if (any(!is.finite(observed))) {
    stop("`observed` contains non-finite values.")
  }
  
  predictor <- make_HG_fit_predictor(
    fit            = fit,
    bayes_fit_spec = bayes_fit_spec
  )
  
  test_prediction <- predictor$predict_raw()
  
  if (length(test_prediction) != length(observed)) {
    stop(
      "Observed and predicted signals have different lengths."
    )
  }
  
  ## --------------------------------------------------------------
  ## Initial sigma from deterministic-fit residuals
  ## ------------------------------------------------------------
  
  if (is.null(sigma_start)) {
    
    sigma_start <- estimate_sigma_from_HG_fit(
      fit,
      method = "rmse_df"
    )
  }
  
  if (
    length(sigma_start) != 1L ||
    !is.finite(sigma_start)   ||
    sigma_start <= 0
  ) {
    stop("`sigma_start` must be positive.")
  }
  
  if (
    length(sigma_bounds_factor) != 2L    ||
    any(!is.finite(sigma_bounds_factor)) ||
    any(sigma_bounds_factor <= 0)        ||
    sigma_bounds_factor[1] >= sigma_bounds_factor[2]
  ) {
    stop(
      "`sigma_bounds_factor` must contain two positive increasing values."
    )
  }
  
  sigma_lower <- sigma_start * sigma_bounds_factor[1]
  
  sigma_upper <- sigma_start * sigma_bounds_factor[2]
  
  ## --------------------------------------------------------------
  ## Augment parameter space by log_sigma
  ## ------------------------------------------------------------
  
  theta0 <- c(
    predictor$theta0,
    log_sigma = log(sigma_start)
  )
  
  lower <- c(
    predictor$lower,
    log_sigma = log(sigma_lower)
  )
  
  upper <- c(
    predictor$upper,
    log_sigma = log(sigma_upper)
  )
  
  parameter_names <- names(theta0)
  
  ## --------------------------------------------------------------
  ## Gaussian iid likelihood
  ## ------------------------------------------------------------
  
  log_likelihood <- function(par) {
    
    par <- unlist(par)
    
    if (is.null(names(par))) {
      
      if (length(par) != length(parameter_names)) {
        return(-Inf)
      }
      
      names(par) <- parameter_names
    }
    
    if (
      any(!parameter_names %in% names(par))
    ) {
      return(-Inf)
    }
    
    par <- par[parameter_names]
    
    if (any(!is.finite(par))) {
      return(-Inf)
    }
    
    theta <- par[predictor$parameter_names]
    
    sigma <- exp(par["log_sigma"])
    
    predicted <- tryCatch(
      predictor$predict_raw(theta),
      error = function(e) NULL
    )
    
    if (
      is.null(predicted)                    ||
      length(predicted) != length(observed) ||
      any(!is.finite(predicted))            ||
      !is.finite(sigma)                     ||
      sigma <= 0
    ) {
      return(-Inf)
    }
    
    sum(
      stats::dnorm(
        observed,
        mean = predicted,
        sd   = sigma,
        log  = TRUE
      )
    )
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
        observed     = observed,
        sigma_start  = sigma_start,
        sigma_bounds = c(
          lower = sigma_lower,
          upper = sigma_upper
        ),
        bayes_theta0 = theta0,
        bayes_lower  = lower,
        bayes_upper  = upper,
        bayes_parameter_names = parameter_names,
        log_likelihood        = log_likelihood,
        prior = prior,
        setup = setup
      )
    ),
    class = c(
      "HG_bayes_model",
      class(predictor)
    )
  )
}

make_bounded_cloud <- function(
    center,
    lower,
    upper,
    n,
    z_sd = 0.10
) {
  
  center          <- unlist(center)
  parameter_names <- names(center)
  lower <- lower[parameter_names]
  upper <- upper[parameter_names]
  
  if (
    any(!is.finite(center)) ||
    any(!is.finite(lower))  ||
    any(!is.finite(upper))
  ) {
    stop(
      "center, lower and upper must all be finite."
    )
  }
  
  if (any(lower >= upper)) {
    stop("All lower bounds must be below upper bounds.")
  }
  
  if (length(z_sd) == 1L) {
    z_sd <- rep(
      z_sd,
      length(center)
    )
  }
  
  if (
    length(z_sd) != length(center)
  ) {
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
    
    center_logit  <- qlogis(relative_position)
    sampled_logit <- center_logit + rnorm(n, mean = 0, sd = z_sd[j])
    result[, j]   <- lower[j] + (upper[j] - lower[j]) * plogis(sampled_logit)
  }
  
  result
}

make_HG_bayes_populations <- function(
    bayes_model,
    n_internal_chains = 12L,
    n_Z               = 200L,
    start_sd          = 0.08,
    Z_sd              = 0.12,
    seed              = 0
) {
  
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  start_value <- make_bounded_cloud(
    center = bayes_model$bayes_theta0,
    lower  = bayes_model$bayes_lower,
    upper  = bayes_model$bayes_upper,
    n      = n_internal_chains,
    z_sd   = start_sd
  )
  
  Z <- make_bounded_cloud(
    center = bayes_model$bayes_theta0,
    lower  = bayes_model$bayes_lower,
    upper  = bayes_model$bayes_upper,
    n      = n_Z,
    z_sd   = Z_sd
  )
  
  ## Exact BOBYQA-derived start included.
  start_value[1L, ] <- bayes_model$bayes_theta0
  
  Z[1L, ] <- bayes_model$bayes_theta0
  
  list(
    startValue = start_value,
    Z          = Z
  )
}

## Nonlinear optim + Bayesian inference high-level wrapper ---------------------

fit_HG_nloptr_MCMC <- function(
    fit_data,
    fixed_params,
    fit_spec,
    signal_col = "signal",
    
    ## ---- NLOPTR -----------------------------------------------
    weights       = NULL,
    nloptr_method = c(
      "nloptr",
      "minpack.lm"
    ),
    nloptr_algorithm = c(
      "NLOPT_LN_BOBYQA",
      "NLOPT_LN_SBPLX",
      "NLOPT_LN_NELDERMEAD"
    ),
    loss = c(
      "sse",
      "mae",
      "huber"
    ),
    huber_delta     = 1,
    penalty_fail    = 1e100,
    nloptr_control  = list(
      maxeval     = 200,
      xtol_rel    = 1e-5,
      print_level = 1
    ),
    minpack_control = list(
      maxiter = 100,
      maxfev  = 1000,
      nprint  = 1
    ),
    checkpoint_path = NULL,
    
    ## ---- Automatic Bayesian bounds ----------------------------
    bayes_fit_spec = NULL,
    
    logB_half_width = 2,
    
    response_rel_half_width = 2,
    response_map_half_width = 1,
    
    clip_to_original_bounds = TRUE,
    
    ## ---- Sigma ------------------------------------------------
    sigma_method = c(
      "rmse_df",
      "rmse",
      "mad"
    ),
    
    sigma_bounds_factor = c(
      0.1,
      10
    ),
    
    ## ---- DEzs -------------------------------------------------
    iterations = 24000L,
    burnin     = 6000L,
    thin       = 1L,
    
    n_internal_chains = 12L,
    n_Z               = 200L,
    
    start_sd = 0.08,
    Z_sd     = 0.12,
    
    seed = 0,
    
    consoleUpdates = 1000L,
    
    mcmc_extra_settings = list(),
    
    verbose = TRUE
) {
  
  sigma_method <- match.arg(sigma_method)
  
  ## ============================================================
  ## 1. Deterministic NLOPTR fit
  ## ============================================================
  
  fit <- fit_HG_titration(
    fit_data         = fit_data,
    fixed_params     = fixed_params,
    fit_spec         = fit_spec,
    signal_col       = signal_col,
    weights          = weights,
    method           = nloptr_method,
    nloptr_algorithm = nloptr_algorithm,
    loss             = loss,
    huber_delta      = huber_delta,
    penalty_fail     = penalty_fail,
    nloptr_control   = nloptr_control,
    minpack_control  = minpack_control,
    verbose          = verbose,
    checkpoint_path  = checkpoint_path
  )
  
  ## ============================================================
  ## 2. Build finite Bayesian parameter domain
  ## ============================================================
  
  if (is.null(bayes_fit_spec)) {
    
    bayes_fit_spec <- make_HG_bayes_fit_spec(
        fit                     = fit,
        logB_half_width         = logB_half_width,
        response_rel_half_width = response_rel_half_width,
        response_map_half_width = response_map_half_width,
        clip_to_original_bounds = clip_to_original_bounds
      )
  }
  
  ## ============================================================
  ## 3. Sigma initialization
  ## ============================================================
  
  sigma_start <- estimate_sigma_from_HG_fit(
      fit,
      method = sigma_method
    )
  
  ## ============================================================
  ## 4. Bayesian model
  ## ============================================================
  
  observed <- as.numeric(
    fit_data[[signal_col]]
  )
  
  bayes_model <- make_HG_bayes_model(
      fit                 = fit,
      observed            = observed,
      bayes_fit_spec      = bayes_fit_spec,
      sigma_start         = sigma_start,
      sigma_bounds_factor = sigma_bounds_factor
    )
  
  ## ============================================================
  ## 5. DEzs starting populations
  ## ============================================================
  
  populations <- make_HG_bayes_populations(
      bayes_model       = bayes_model,
      n_internal_chains = n_internal_chains,
      n_Z               = n_Z,
      start_sd          = start_sd,
      Z_sd              = Z_sd,
      seed              = seed
    )
  
  ## ============================================================
  ## 6. MCMC
  ## ============================================================
  
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
  
  ## ============================================================
  ## 7. Posterior
  ## ============================================================
  
  posterior <- as.data.frame(
    BayesianTools::getSample(
      bayes,
      parametersOnly = TRUE
    )
  )
  
  ## Physical sigma as derived parameter
  posterior$sigma <- exp(posterior$log_sigma)
  
  chemical_parameters <- bayes_model$parameter_names
  
  posterior_parameters <- posterior[
      ,
      c(
        chemical_parameters,
        "sigma"
      ),
      drop = FALSE
    ]
  
  posterior_stats <- posterior_summary(
      posterior_parameters
    )
  
  ## ============================================================
  ## 8. Return
  ## ============================================================
  
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
  
  class(out) <-
    "HG_BOBYQA_MCMC_fit"
  
  out
}

## Helper function for extracting summary
posterior_summary <- function(
    posterior,
    conf_level = 0.95
) {
  alpha <- 1 - conf_level
  
  data.frame(
    parameter = names(posterior),
    
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
      sd,
      numeric(1)
    ),
    
    lower = vapply(
      posterior,
      quantile,
      numeric(1),
      probs = alpha / 2,
      names = FALSE
    ),
    
    upper = vapply(
      posterior,
      quantile,
      numeric(1),
      probs = 1 - alpha / 2,
      names = FALSE
    ),
    
    row.names = NULL
  )
}

## Example run -----------------------------------------------------------------

if (FALSE) { ### turn TRUE to run example
  estimate_sigma_from_HG_fit(
    fit1,
    method = "rmse_df"
  )
  
  make_HG_bayes_fit_spec(
    fit1,
    logB_half_width         = 2,
    response_rel_half_width = 2,
    response_map_half_width = 1
  )
  
  fit_bayes1 <- fit_HG_nloptr_MCMC(
    fit_data                = fit_data,
    fixed_params            = fixed_params,
    fit_spec                = fit_spec,
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
  
  fit_bayes1$fit$par
  fit_bayes1$posterior_summary
  
  BayesianTools::tracePlot(
    fit_bayes1$bayes,
    parametersOnly = TRUE
  )
  
  BayesianTools::marginalPlot(
    fit_bayes1$bayes
  )
  
  BayesianTools::correlationPlot(
    fit_bayes1$bayes,
    parametersOnly = TRUE
  )
}
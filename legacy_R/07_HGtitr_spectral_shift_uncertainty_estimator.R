################################################################################
## 07_HGtitr_spectral_shift_uncertainty_estimator.R
##
## Bayesian uncertainty analysis for HG spectral-shift fits.
##
## Requires:
##   source("01_HGtitr_simulator.R")
##   source("02_HGtitr_fitter.R")
##   source("04_HGtitr_multichannel_fitter.R")
##   source("06_HGtitr_spectral_shift_fitter_species_specific.R")
##
## Package:
##   BayesianTools
##
## Main idea
## ---------
## The nonlinear parameters from the converged spectral-shift fit are sampled:
##
##   - logB_* parameters
##   - delta_lambda_* parameters
##   - logC50_shift / hill_shift, if present
##   - residual noise parameter(s)
##
## At every likelihood evaluation the smooth B-spline response spectra are
## re-profiled by linear least squares through HG_eval_spectral_shift().
##
## This is therefore a PROFILED Bayesian analysis: uncertainty of the nonlinear
## equilibrium / shift parameters is sampled, whereas conditional uncertainty of
## the B-spline coefficients is not sampled explicitly.
##
## Two residual likelihoods are provided:
##
##   "iid"  : independent Gaussian residuals
##   "ar1"  : Gaussian AR(1) correlation across wavelength within each spectrum
##
## For densely sampled spectra, "ar1" is generally preferable because adjacent
## wavelengths are strongly correlated and should not be treated as independent.
################################################################################


## ---------------------------------------------------------------------------
## Basic helpers
## ---------------------------------------------------------------------------

HG_spectral_estimate_sigma <- function(
    fit
) {
  r <- fit$prediction$residual

  r <- r[is.finite(r)]

  if (length(r) < 2L) {
    stop("Not enough finite residuals to estimate sigma.")
  }

  stats::sd(r)
}


HG_spectral_estimate_rho <- function(
    fit,
    clamp = c(-0.95, 0.995)
) {
  dat <- fit$prediction

  dat <- dat[
    order(
      dat$point_id,
      dat$wavelength
    ),
    ,
    drop = FALSE
  ]

  num <- 0
  den <- 0

  for (id in unique(dat$point_id)) {

    r <- dat$residual[dat$point_id == id]

    r <- r[is.finite(r)]

    if (length(r) < 3L) {
      next
    }

    r <- r - mean(r)

    num <- num + sum(r[-1L] * r[-length(r)])

    den <- den + sum(r[-length(r)]^2)
  }

  rho <- if (
    is.finite(den) &&
    den > 0
  ) {
    num / den
  } else {
    0
  }

  rho <- pmin(
    pmax(
      rho,
      clamp[1L]
    ),
    clamp[2L]
  )

  as.numeric(rho)
}


HG_spectral_supported_shift <- function(
    fit
) {
  wave <- sort(
    unique(
      fit$prediction$wavelength
    )
  )

  boundary <- fit$basis_spec$Boundary.knots

  left_margin <- min(wave) - boundary[1L]

  right_margin <- boundary[2L] - max(wave)

  max(
    0,
    min(
      left_margin,
      right_margin
    )
  )
}


## ---------------------------------------------------------------------------
## Default prior ranges around converged deterministic solution
## ---------------------------------------------------------------------------

make_HG_spectral_shift_bayes_bounds <- function(
    fit,
    logB_width           = 1.5,
    delta_relative_width = 0.75,
    delta_min_width      = 25,
    logC50_width         = 1.5,
    hill_relative_width  = 0.75,
    sigma_factor         = 5,
    rho_bounds = c(
      -0.50,
       0.999
    ),
    noise_model = c(
      "iid",
      "ar1"
    ),
    lower_override = NULL,
    upper_override = NULL
) {
  noise_model <- match.arg(noise_model)

  p <- fit$par

  lower <- upper <- p

  for (nm in names(p)) {

    value <- unname(p[nm])

    if (grepl("^logB_", nm)) {

      lower[nm] <- value - logB_width
      upper[nm] <- value + logB_width

    } else if (grepl("^delta_lambda_", nm)) {

      w <- max(
        delta_min_width,
        abs(value) * delta_relative_width
      )

      lower[nm] <- value - w
      upper[nm] <- value + w

    } else if (identical(nm, "logC50_shift")) {

      lower[nm] <- value - logC50_width
      upper[nm] <- value + logC50_width

    } else if (identical(nm, "hill_shift")) {

      w <- max(
        0.5,
        abs(value) * hill_relative_width
      )

      lower[nm] <- max(
        0.05,
        value - w
      )

      upper[nm] <- value + w

    } else {

      w <- max(
        1,
        abs(value)
      )

      lower[nm] <- value - w
      upper[nm] <- value + w
    }
  }

  ## User overrides first -----------------------------------------------------
  if (!is.null(lower_override)) {

    common <- intersect(
      names(lower_override),
      names(lower)
    )

    lower[common] <- lower_override[common]
  }

  if (!is.null(upper_override)) {

    common <- intersect(
      names(upper_override),
      names(upper)
    )

    upper[common] <- upper_override[common]
  }

  ## Then intersect delta priors with the wavelength support actually present
  ## in the deterministic B-spline basis.
  supported_shift <- HG_spectral_supported_shift(fit)

  delta_names <- grep(
    "^delta_lambda_",
    names(p),
    value = TRUE
  )

  if (
    length(delta_names) > 0L   &&
    is.finite(supported_shift) &&
    supported_shift > 0
  ) {

    lower[delta_names] <- pmax(
      lower[delta_names],
      -supported_shift
    )

    upper[delta_names] <- pmin(
      upper[delta_names],
      supported_shift
    )
  }

  sigma_start <- HG_spectral_estimate_sigma(fit)

  if (
    !is.finite(sigma_start) ||
    sigma_start <= 0
  ) {
    stop("Could not estimate a positive residual sigma.")
  }

  start <- c(
    p,
    log_sigma = log(sigma_start)
  )

  lower <- c(
    lower,
    log_sigma = log(sigma_start / sigma_factor)
  )

  upper <- c(
    upper,
    log_sigma = log(sigma_start * sigma_factor)
  )

  if (noise_model == "ar1") {

    if (
      length(rho_bounds) != 2L    ||
      any(!is.finite(rho_bounds)) ||
      rho_bounds[1L] <= -1        ||
      rho_bounds[2L] >= 1         ||
      rho_bounds[1L] >= rho_bounds[2L]
    ) {
      stop("`rho_bounds` must satisfy -1 < lower < upper < 1.")
    }

    rho_raw <- HG_spectral_estimate_rho(
      fit,
      clamp = c(-0.999999, 0.999999)
    )

    ## Keep the starting rho strictly inside the requested prior bounds.
    rho_margin <- max(
      1e-6,
      1e-4 * diff(rho_bounds)
    )

    rho_start <- min(
      max(
        rho_raw,
        rho_bounds[1L] + rho_margin
      ),
      rho_bounds[2L] - rho_margin
    )

    start <- c(
      start,
      rho_lambda = rho_start
    )

    lower <- c(
      lower,
      rho_lambda = rho_bounds[1L]
    )

    upper <- c(
      upper,
      rho_lambda = rho_bounds[2L]
    )
  }

  ## Validate interval geometry before nudging boundary-equal starts.
  if (
    any(!is.finite(start)) ||
    any(!is.finite(lower)) ||
    any(!is.finite(upper)) ||
    any(lower >= upper)
  ) {
    stop(
      "Invalid Bayesian start/lower/upper bounds: non-finite values or ",
      "lower >= upper."
    )
  }

  ## A deterministic optimum may lie exactly on a user/prior boundary.
  ## BayesianTools requires a strictly interior start, so equality is nudged
  ## inward. A start genuinely outside the interval remains an error.
  if (
    any(start < lower) ||
    any(start > upper)
  ) {

    bad <- names(start)[
      start < lower |
      start > upper
    ]

    stop(
      "Deterministic start lies outside Bayesian bounds for: ",
      paste(
        bad,
        collapse = ", "
      ),
      ". Widen the corresponding prior bounds."
    )
  }

  margin <- pmax(
    1e-10,
    1e-8 * (upper - lower)
  )

  start <- pmin(
    pmax(
      start,
      lower + margin
    ),
    upper - margin
  )

  list(
    start = start,
    lower = lower,
    upper = upper,
    sigma_start = sigma_start,
    rho_start = if (
      noise_model == "ar1"
    ) {
      unname(
        start["rho_lambda"]
      )
    } else {
      NULL
    },
    supported_shift = supported_shift
  )
}


## ---------------------------------------------------------------------------
## Residual log-likelihoods
## ---------------------------------------------------------------------------

HG_loglik_iid <- function(
    residual,
    sigma
) {
  if (
    !is.finite(sigma) ||
    sigma <= 0
  ) {
    return(
      -Inf
    )
  }

  sum(
    stats::dnorm(
      residual,
      mean = 0,
      sd   = sigma,
      log  = TRUE
    )
  )
}


HG_loglik_ar1_by_spectrum <- function(
    residual,
    point_id,
    sigma,
    rho
) {
  if (
    !is.finite(sigma) ||
    sigma <= 0        ||
    !is.finite(rho)   ||
    abs(rho) >= 1
  ) {
    return(
      -Inf
    )
  }

  innovation_sd <- sigma * sqrt(1 - rho^2)

  out <- 0

  ids <- unique(point_id)

  for (id in ids) {

    e <- residual[point_id == id]

    if (
      length(e) == 0L ||
      any(!is.finite(e))
    ) {
      return(
        -Inf
      )
    }

    ## Stationary marginal distribution for first wavelength.
    out <- out +
      stats::dnorm(
        e[1L],
        mean = 0,
        sd   = sigma,
        log  = TRUE
      )

    if (length(e) > 1L) {

      innovations <- e[-1L] - rho * e[-length(e)]

      out <- out +
        sum(
          stats::dnorm(
            innovations,
            mean = 0,
            sd   = innovation_sd,
            log  = TRUE
          )
        )
    }
  }

  out
}


## ---------------------------------------------------------------------------
## Build profiled spectral-shift likelihood
## ---------------------------------------------------------------------------

make_HG_spectral_shift_bayes_model <- function(
    fit,
    noise_model        = c("iid", "ar1"),
    weights            = NULL,
    endpoint_tol       = 1e-7,
    endpoint_maxiter   = 500L,
    endpoint_damping   = 0.5,
    use_optim_fallback = TRUE
) {
  noise_model <- match.arg(noise_model)

  if (
    !inherits(
      fit,
      "HG_spectral_shift_fit"
    )
  ) {
    stop("`fit` must be an HG_spectral_shift_fit.")
  }

  spectral_data <- prepare_HG_spectral_data(
      fit$data_used,
      signal_col = "signal"
    )

  nonlinear_names <- names(fit$par)

  likelihood <- function(
      theta
  ) {

    theta <- as.numeric(theta)

    if (
      length(theta) !=
      length(
        c(
          nonlinear_names,
          "log_sigma",
          if (
            noise_model == "ar1"
          ) {
            "rho_lambda"
          }
        )
      )
    ) {
      return(
        -Inf
      )
    }

    parameter_names <- c(
      nonlinear_names,
      "log_sigma",
      if (
        noise_model == "ar1"
      ) {
        "rho_lambda"
      }
    )

    names(theta) <- parameter_names

    nonlinear_par <- theta[nonlinear_names]

    sigma <- exp(theta["log_sigma"])

    eval <- tryCatch(
      HG_eval_spectral_shift(
        par                = nonlinear_par,
        spectral_data      = spectral_data,
        species_vec        = fit$species_vec,
        basis_spec         = fit$basis_spec,
        shifted_species    = fit$shifted_species,
        shift_mode         = fit$shift_mode,
        weights            = weights,
        ridge              = fit$ridge,
        endpoint_tol       = endpoint_tol,
        endpoint_maxiter   = endpoint_maxiter,
        endpoint_damping   = endpoint_damping,
        use_optim_fallback = use_optim_fallback
      ),
      error = function(e) {
        NULL
      }
    )

    if (
      is.null(eval) ||
      any(!is.finite(eval$residual))
    ) {
      return(
        -Inf
      )
    }

    if (noise_model == "iid") {

      return(
        HG_loglik_iid(residual = eval$residual, sigma = sigma)
      )
    }

    rho <- unname(theta["rho_lambda"])

    HG_loglik_ar1_by_spectrum(
      residual = eval$residual,
      point_id = spectral_data$data$point_id,
      sigma    = sigma,
      rho      = rho
    )
  }

  predictor <- function(
      theta
  ) {

    theta <- as.numeric(theta)

    parameter_names <- c(
      nonlinear_names,
      "log_sigma",
      if (
        noise_model == "ar1"
      ) {
        "rho_lambda"
      }
    )

    names(theta) <- parameter_names

    eval <- HG_eval_spectral_shift(
        par                = theta[nonlinear_names],
        spectral_data      = spectral_data,
        species_vec        = fit$species_vec,
        basis_spec         = fit$basis_spec,
        shifted_species    = fit$shifted_species,
        shift_mode         = fit$shift_mode,
        weights            = weights,
        ridge              = fit$ridge,
        endpoint_tol       = endpoint_tol,
        endpoint_maxiter   = endpoint_maxiter,
        endpoint_damping   = endpoint_damping,
        use_optim_fallback = use_optim_fallback
      )

    eval$predicted
  }

  list(
    likelihood      = likelihood,
    predictor       = predictor,
    spectral_data   = spectral_data,
    nonlinear_names = nonlinear_names,
    noise_model     = noise_model
  )
}

## ---------------------------------------------------------------------------
## Starting cloud around deterministic optimum
## ---------------------------------------------------------------------------

make_HG_spectral_shift_start_population <- function(
    start,
    lower,
    upper,
    n_internal_chains = NULL,
    jitter_fraction   = 0.01,
    seed              = NULL
) {
  if (!is.null(seed)) {
    set.seed(seed)
  }

  parameter_names <- names(start)

  start <- as.numeric(start)
  lower <- as.numeric(lower)
  upper <- as.numeric(upper)
  n_par <- length(start)

  if (is.null(n_internal_chains)) {
    n_internal_chains <- max(
      12L,
      3L * n_par
    )
  }

  n_internal_chains <- as.integer(n_internal_chains)

  span <- upper - lower

  pop <- matrix(
    NA_real_,
    nrow = n_internal_chains,
    ncol = n_par
  )

  for (i in seq_len(n_internal_chains)) {

    candidate <- start +
      stats::rnorm(
        n_par,
        mean = 0,
        sd   = jitter_fraction * span
      )

    eps <- 1e-8 * pmax(
        span,
        1
      )

    candidate <- pmin(
      pmax(
        candidate,
        lower + eps
      ),
      upper - eps
    )

    pop[i, ] <- candidate
  }

  ## Put deterministic optimum into first internal chain.
  pop[1L, ] <- start

  colnames(pop) <- parameter_names

  pop
}

## ---------------------------------------------------------------------------
## Posterior summary helper
## ---------------------------------------------------------------------------

posterior_summary_HG_spectral_shift <- function(
    mcmc,
    probs = c(0.025, 0.975)
) {
  ## BayesianTools::DEzs does not record iterations designated as burn-in.
  ## Therefore the returned chain already excludes burn-in.
  sample <-BayesianTools::getSample(
      mcmc,
      parametersOnly = TRUE
    )

  sample <- as.matrix(sample)

  out <- data.frame(
    parameter = colnames(
        sample
      ),

    mean = apply(
        sample,
        2,
        mean
      ),

    median = apply(
        sample,
        2,
        stats::median
      ),

    sd = apply(
        sample,
        2,
        stats::sd
      ),

    lower = apply(
        sample,
        2,
        stats::quantile,
        probs =
          probs[1L],
        names = FALSE
      ),

    upper = apply(
        sample,
        2,
        stats::quantile,
        probs =
          probs[2L],
        names = FALSE
      ),

    stringsAsFactors = FALSE
  )

  ## Also report sigma on its natural scale.
  if (
    "log_sigma" %in%
    out$parameter
  ) {

    sigma_sample <- exp(
        sample[
          ,
          "log_sigma"
        ]
      )

    sigma_row <- data.frame(
      parameter = "sigma",

      mean = mean(
          sigma_sample
        ),

      median = stats::median(
          sigma_sample
        ),

      sd = stats::sd(
          sigma_sample
        ),

      lower = stats::quantile(
          sigma_sample,
          probs = probs[1L],
          names = FALSE
        ),

      upper = stats::quantile(
          sigma_sample,
          probs = probs[2L],
          names = FALSE
        ),

      stringsAsFactors = FALSE
    )

    out <- rbind(
      out,
      sigma_row
    )
  }

  rownames(out) <- NULL

  out
}

## ---------------------------------------------------------------------------
## Main wrapper
## ---------------------------------------------------------------------------

fit_HG_spectral_shift_MCMC <- function(
    fit,

    noise_model = c("ar1", "iid" ),

    logB_width           = 1.5,
    delta_relative_width = 0.75,
    delta_min_width      = 25,
    logC50_width         = 1.5,
    hill_relative_width  = 0.75,
    sigma_factor         = 5,

    rho_bounds = c(-0.50, 0.999),

    lower_override = NULL,
    upper_override = NULL,

    iterations = 12000,
    burnin     = 3000,
    thin       = 1,

    n_internal_chains     = NULL,
    start_jitter_fraction = 0.01,

    sampler = "DEzs",

    seed = 1,

    consoleUpdates = 500,

    weights = NULL,

    endpoint_tol       = 1e-7,
    endpoint_maxiter   = 500L,
    endpoint_damping   = 0.5,
    use_optim_fallback = TRUE,

    de_settings = list(
      f        = 1.0,
      pGamma1  = 0.01,
      pSnooker = 0.01,
      eps.mult = 0.05
    )
) {
  noise_model <- match.arg(noise_model)

  if (!requireNamespace(
    "BayesianTools",
    quietly = TRUE
  )) {
    stop("Package `BayesianTools` is required.")
  }

  set.seed(seed)

  bounds <- make_HG_spectral_shift_bayes_bounds(
      fit                  = fit,
      logB_width           = logB_width,
      delta_relative_width = delta_relative_width,
      delta_min_width      = delta_min_width,
      logC50_width         = logC50_width,
      hill_relative_width  = hill_relative_width,
      sigma_factor         = sigma_factor,
      rho_bounds           = rho_bounds,
      noise_model          = noise_model,
      lower_override       = lower_override,
      upper_override       = upper_override
    )

  model <- make_HG_spectral_shift_bayes_model(
      fit                = fit,
      noise_model        = noise_model,
      weights            = weights,
      endpoint_tol       = endpoint_tol,
      endpoint_maxiter   = endpoint_maxiter,
      endpoint_damping   = endpoint_damping,
      use_optim_fallback = use_optim_fallback
    )

  prior <- BayesianTools::createUniformPrior(
      lower = bounds$lower,
      upper = bounds$upper,
      best  = bounds$start
    )

  bayesianSetup <- BayesianTools::createBayesianSetup(
      likelihood = model$likelihood,
      prior      = prior,
      names      = names(bounds$start)
    )

  start_population <- make_HG_spectral_shift_start_population(
      start             = bounds$start,
      lower             = bounds$lower,
      upper             = bounds$upper,
      n_internal_chains = n_internal_chains,
      jitter_fraction   = start_jitter_fraction,
      seed              = seed
    )

  settings <- utils::modifyList(
    list(
      iterations     = iterations,
      burnin         = burnin,
      thin           = thin,
      startValue     = start_population,
      consoleUpdates = consoleUpdates,
      message        = TRUE
    ),
    de_settings
  )

  mcmc <- BayesianTools::runMCMC(
      bayesianSetup = bayesianSetup,
      sampler       = sampler,
      settings      = settings
    )

  posterior_summary <- posterior_summary_HG_spectral_shift(
      mcmc = mcmc
    )

  out <- list(
    fit               = fit,
    mcmc              = mcmc,
    posterior         = mcmc,
    posterior_summary = posterior_summary,
    bounds            = bounds,
    bayesianSetup     = bayesianSetup,
    bayes_model       = model,
    noise_model       = noise_model,
    start_population  = start_population,
    settings          = settings
  )

  class(out) <-
    "HG_spectral_shift_MCMC_fit"

  out
}

# Helper for extracting model metrics ------------------------------------------

profile_AR1_residual_likelihood <- function(
    prediction,
    rho_bounds = c(-0.5, 0.9999)
) {
  
  dat <- prediction |>
    dplyr::filter(
      is.finite(residual),
      is.finite(wavelength)
    ) |>
    dplyr::arrange(
      point_id,
      wavelength
    )
  
  residual_list <- split(
    dat$residual,
    dat$point_id
  )
  
  N <- sum(lengths(residual_list))
  
  calc_at_rho <- function(rho) {
    
    if (!is.finite(rho) || abs(rho) >= 1) {
      return(
        list(
          logLik = -Inf
        )
      )
    }
    
    one_minus_rho2 <- 1 - rho^2
    
    Q <- 0
    logdet_R <- 0
    
    for (e in residual_list) {
      
      m <- length(e)
      
      if (m == 0L) {
        next
      }
      
      ## Stationary first observation
      Q <- Q + e[1]^2
      
      if (m > 1L) {
        
        innovations <- e[-1L] - rho * e[-m]
        
        Q <- Q + sum(innovations^2) / one_minus_rho2
        
        ## determinant of AR(1) correlation matrix
        logdet_R <- logdet_R + (m - 1) * log(one_minus_rho2)
      }
    }
    
    sigma2_ML <- Q / N
    
    if (
      !is.finite(sigma2_ML) ||
      sigma2_ML <= 0
    ) {
      return(
        list(
          logLik = -Inf
        )
      )
    }
    
    logLik <- -0.5 * (N * (log(2 * pi) + 1 + log(sigma2_ML)) + logdet_R
    )
    
    list(
      logLik   = logLik,
      sigma2   = sigma2_ML,
      Q        = Q,
      logdet_R = logdet_R
    )
  }
  
  
  opt <- optimize(
    f = function(rho) {
      -calc_at_rho(rho)$logLik
    },
    interval = rho_bounds,
    maximum  = FALSE
  )
  
  rho_ML <- opt$minimum
  
  z <- calc_at_rho(rho_ML)
  
  list(
    N         = N,
    n_spectra = length(residual_list),
    rho_ML    = rho_ML,
    sigma_ML  = sqrt(z$sigma2),
    logLik    = z$logLik,
    Q         = z$Q,
    logdet_R  = z$logdet_R
  )
}

extract_spectral_fit_metrics_AR1 <- function(
    fit,
    model_name = NULL,
    rho_bounds = c(-0.5, 0.9999)
) {
  
  ar1 <- profile_AR1_residual_likelihood(
    prediction = fit$prediction,
    rho_bounds = rho_bounds
  )
  
  N <- ar1$N
  
  p_nonlinear <- length(fit$par)
  
  p_spectral <- fit$linear_rank
  
  ## +1 sigma
  ## +1 rho_lambda
  k <- p_nonlinear + p_spectral + 2L
  
  logLik <- ar1$logLik
  
  AIC <- -2 * logLik + 2 * k
  
  ## See caveat below: approximate small-sample correction
  AICc <- AIC + 2 * k * (k + 1) / (N - k - 1)
  
  BIC <- -2 * logLik + log(N) * k
  
  tibble::tibble(
    model = model_name,
    
    N = N,
    n_spectra = ar1$n_spectra,
    
    p_nonlinear = p_nonlinear,
    p_spectral  = p_spectral,
    k           = k,
    
    SSE      = fit$objective_value,
    RMSE_iid = sqrt(fit$objective_value / nrow(fit$prediction)),
    
    rho_AR1   = ar1$rho_ML,
    sigma_AR1 = ar1$sigma_ML,
    
    logLik_AR1 = logLik,
    
    AIC_AR1         = AIC,
    AICc_AR1_approx = AICc,
    BIC_AR1         = BIC
  )
}

################################################################################
## Example: biliverdin species-specific spectral-shift fit
################################################################################

if (FALSE) { ### turn TRUE to run example

  source("01_HGtitr_simulator.R")
  source("02_HGtitr_fitter.R")
  source("04_HGtitr_multichannel_fitter.R")
  source("06_HGtitr_spectral_shift_fitter_species_specific.R")
  source("07_HGtitr_spectral_shift_uncertainty_estimator.R")


  ## Load the best deterministic solution from the multistart search.
  fit_bvd_shift_species <- readRDS(
    paste0(
      "HG_fits_biliverdin_species_specific_shift/",
      "spectral_shift_021.rds"
    )
  )


  ## --------------------------------------------------------------------------
  ## Recommended final analysis:
  ## AR(1) residual correlation along wavelength within each titration spectrum.
  ## --------------------------------------------------------------------------

  fit_bvd_shift_bayes <- fit_HG_spectral_shift_MCMC(
    fit = fit_bvd_shift_species,

    noise_model = "ar1",

    ## The defaults construct priors around the converged deterministic optimum.
    ## Explicit overrides can be supplied if desired.
    lower_override = c(
      logB_H1G1 = 2,
      logB_H1G2 = 6,
      delta_lambda_H1G1 = -150,
      delta_lambda_H1G2 = -150
    ),

    upper_override = c(
      logB_H1G1 = 8,
      logB_H1G2 = 14,
      delta_lambda_H1G1 = 150,
      delta_lambda_H1G2 = 150
    ),

    iterations = 12000,

    burnin = 3000,

    thin = 1,

    n_internal_chains = 18,

    start_jitter_fraction = 0.01,

    seed = 123,

    consoleUpdates = 500
  )


  fit_bvd_shift_bayes$posterior_summary


  ## Standard BayesianTools diagnostics.
  BayesianTools::tracePlot(
    fit_bvd_shift_bayes$mcmc
  )

  BayesianTools::marginalPlot(
    fit_bvd_shift_bayes$mcmc
  )

  BayesianTools::correlationPlot(
    fit_bvd_shift_bayes$mcmc
  )


  ## Extract posterior sample for custom ggplot figures.
  ## Burn-in specified in runMCMC() is already omitted from the DEzs chain.
  sample_bvd_shift <- BayesianTools::getSample(
      fit_bvd_shift_bayes$mcmc,
      parametersOnly = TRUE
    )

  sample_bvd_shift <- as.data.frame(
      sample_bvd_shift
    )

  head(sample_bvd_shift)
}

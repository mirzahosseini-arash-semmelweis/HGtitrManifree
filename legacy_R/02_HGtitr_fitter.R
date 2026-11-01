################################################################################
## Fitting of Host-Guest titration data
################################################################################

## Parameter-table helper functions for fitting --------------------------------

## fit_spec format
## fit_spec <- list(
## 
##   logB_vec = list(
##     start = c(   5,    8),
##     lower = c(-Inf, -Inf),
##     upper = c( Inf,  Inf),
##     fit   = c(TRUE, TRUE),
## 
##     ## Characteristic step size used only for mapping
##     ## infinite parameter ranges onto [0,1].
##     map_scale = c(2, 2)
##   ),
## 
##   response_vec = list(
##     start = c(0, 0,  2e5,  3e5),
##     lower = c(0, 0, -Inf, -Inf),
##     upper = c(0, 0,  Inf,  Inf),
## 
##     ## For example, keep free H and free G responses fixed at zero.
##     fit = c(FALSE, FALSE, TRUE, TRUE),
## 
##     map_scale = c(NA, NA, 2e5, 3e5)
##   )
## )

prepare_HG_fit_spec <- function(
    fit_spec,
    species_vec
) {
  
  stopifnot(
    is.list(fit_spec),
    is.character(species_vec),
    length(species_vec) >= 1L
  )
  
  required_groups <- c(
    "logB_vec",
    "response_vec"
  )
  
  missing_groups <- setdiff(
    required_groups,
    names(fit_spec)
  )
  
  if (length(missing_groups)) {
    stop(
      "Missing fit specification group(s): ",
      paste(missing_groups, collapse = ", ")
    )
  }
  
  group_labels <- list(
    logB_vec     = species_vec,
    response_vec = c(
      "H1G0",
      "H0G1",
      species_vec
    )
  )
  
  normalized_spec <- list()
  table_rows      <- list()
  
  recycle_to_length <- function(
    x,
    n,
    name
  ) {
    
    if (length(x) == 1L) {
      return(rep(x, n))
    }
    
    if (length(x) != n) {
      stop(
        sprintf(
          "`%s` must have length 1 or %d.",
          name,
          n
        )
      )
    }
    
    x
  }
  
  for (group in required_groups) {
    
    x <- fit_spec[[group]]
    
    if (!is.list(x)) {
      stop(
        sprintf(
          "`fit_spec$%s` must be a list.",
          group
        )
      )
    }
    
    required <- c(
      "start",
      "lower",
      "upper"
    )
    
    missing <- setdiff(
      required,
      names(x)
    )
    
    if (length(missing)) {
      stop(
        sprintf(
          "`fit_spec$%s` is missing: %s.",
          group,
          paste(missing, collapse = ", ")
        )
      )
    }
    
    labels <- group_labels[[group]]
    n      <- length(labels)
    
    start <- as.numeric(
      recycle_to_length(
        x$start,
        n,
        paste0(group, "$start")
      )
    )
    
    lower <- as.numeric(
      recycle_to_length(
        x$lower,
        n,
        paste0(group, "$lower")
      )
    )
    
    upper <- as.numeric(
      recycle_to_length(
        x$upper,
        n,
        paste0(group, "$upper")
      )
    )
    
    fit <- if (is.null(x$fit)) {
      rep(TRUE, n)
    } else {
      as.logical(
        recycle_to_length(
          x$fit,
          n,
          paste0(group, "$fit")
        )
      )
    }
    
    transform <- if (is.null(x$transform)) {
      rep("linear", n)
    } else {
      as.character(
        recycle_to_length(
          x$transform,
          n,
          paste0(group, "$transform")
        )
      )
    }
    
    map_scale <- if (is.null(x$map_scale)) {
      rep(NA_real_, n)
    } else {
      as.numeric(
        recycle_to_length(
          x$map_scale,
          n,
          paste0(group, "$map_scale")
        )
      )
    }
    
    if (any(!is.finite(start))) {
      stop(
        sprintf(
          "All `%s$start` values must be finite.",
          group
        )
      )
    }
    
    if (
      any(is.na(lower))  ||
      any(is.nan(lower)) ||
      any(is.na(upper))  ||
      any(is.nan(upper))
    ) {
      stop(
        sprintf(
          "`%s$lower` and `$upper` may be infinite but not NA/NaN.",
          group
        )
      )
    }
    
    if (any(start < lower | start > upper)) {
      stop(
        sprintf(
          "Every `%s$start` must lie within its bounds.",
          group
        )
      )
    }
    
    ## Free parameters require an actual interval.
    if (any(fit & lower >= upper)) {
      stop(
        sprintf(
          "Fitted `%s` parameters require lower < upper.",
          group
        )
      )
    }
    
    ## Fixed parameters may use lower == upper == start.
    if (any(!fit & lower > upper)) {
      stop(
        sprintf(
          "Invalid bounds in fixed `%s` parameters.",
          group
        )
      )
    }
    
    if (
      any(!transform %in% c("linear", "log"))
    ) {
      stop(
        "`transform` must be 'linear' or 'log'."
      )
    }
    
    ## Log interpolation is only implemented for a finite,
    ## strictly positive interval.
    bad_log <- transform == "log" & (
      !is.finite(lower) |
      !is.finite(upper) |
      lower <= 0        |
      upper <= 0
    )
    
    if (any(bad_log)) {
      stop(
        "`transform = 'log'` requires finite positive bounds."
      )
    }
    
    ## ------------------------------------------------------------
    ## Mapping scale for semi-/fully-unbounded parameters
    ## ------------------------------------------------------------
    
    for (i in seq_len(n)) {
      
      if (!fit[i]) {
        next
      }
      
      if (
        !is.finite(lower[i]) ||
        !is.finite(upper[i])
      ) {
        
        if (
          !is.finite(map_scale[i]) ||
          map_scale[i] <= 0
        ) {
          
          map_scale[i] <- if (
            is.finite(lower[i])
          ) {
            
            max(
              abs(start[i] - lower[i]),
              abs(start[i]),
              1
            )
            
          } else if (
            is.finite(upper[i])
          ) {
            
            max(
              abs(upper[i] - start[i]),
              abs(start[i]),
              1
            )
            
          } else {
            
            max(
              abs(start[i]),
              1
            )
          }
        }
      }
    }
    
    normalized_spec[[group]] <- list(
      start     = start,
      lower     = lower,
      upper     = upper,
      fit       = fit,
      transform = transform,
      map_scale = map_scale,
      labels    = labels
    )
    
    fitted_indices <- which(fit)
    
    if (length(fitted_indices)) {
      
      parameter_names <- if (group == "logB_vec") {
        paste0(
          "logB_",
          labels[fitted_indices]
        )
      } else {
        paste0(
          "response_",
          labels[fitted_indices]
        )
      }
      
      table_rows[[group]] <- data.frame(
        name       = parameter_names,
        group      = group,
        index      = fitted_indices,
        species    = labels[fitted_indices],
        lower      = lower[fitted_indices],
        upper      = upper[fitted_indices],
        start      = start[fitted_indices],
        transform  = transform[fitted_indices],
        map_scale  = map_scale[fitted_indices],
        map_center = start[fitted_indices],
        stringsAsFactors = FALSE
      )
    }
  }
  
  par_tbl <- do.call(
    rbind,
    table_rows
  )
  
  rownames(par_tbl) <- NULL
  
  list(
    par_table = par_tbl,
    fit_spec  = normalized_spec
  )
}

par_to_unit <- function(
    par,
    par_tbl
) {
  
  par <- as.numeric(par)
  
  if (length(par) != nrow(par_tbl)) {
    stop("Parameter vector has incorrect length.")
  }
  
  u <- numeric(length(par))
  
  for (i in seq_along(par)) {
    
    x  <- par[i]
    lo <- par_tbl$lower[i]
    hi <- par_tbl$upper[i]
    
    finite_lo <- is.finite(lo)
    finite_hi <- is.finite(hi)
    
    ## ------------------------------------------------------------
    ## Finite lower and upper bounds
    ## ------------------------------------------------------------
    
    if (finite_lo && finite_hi) {
      
      if (par_tbl$transform[i] == "log") {
        u[i] <- (log(x) - log(lo)) / (log(hi) - log(lo))
      } else {
        u[i] <- (x - lo) / (hi - lo)
      }
      
      next
    }
    
    s <- par_tbl$map_scale[i]
    
    ## ------------------------------------------------------------
    ## [lower, +Inf)
    ## ------------------------------------------------------------
    
    if (finite_lo && !finite_hi) {
      d <- max(x - lo, 0)
      u[i] <- d / (s + d)
      
      next
    }
    
    ## ------------------------------------------------------------
    ## (-Inf, upper]
    ## ------------------------------------------------------------
    
    if (!finite_lo && finite_hi) {
      d <- max(hi - x, 0)
      u[i] <- s / (s + d)
      
      next
    }
    
    ## ------------------------------------------------------------
    ## (-Inf, +Inf)
    ## ------------------------------------------------------------
    
    center <- par_tbl$map_center[i]
    
    u[i] <- 0.5 + atan((x - center) / s) / pi
    
  }
  
  names(u) <- par_tbl$name
  
  pmin(pmax(u, 0), 1)
}

unit_to_par <- function(
    u,
    par_tbl,
    eps = 1e-8
) {
  
  u <- as.numeric(u)
  
  if (length(u) != nrow(par_tbl)) {
    stop("Unit parameter vector has incorrect length.")
  }
  
  u <- pmin(pmax(u, 0), 1)
  
  par <- numeric(length(u))
  
  for (i in seq_along(u)) {
    
    z  <- u[i]
    lo <- par_tbl$lower[i]
    hi <- par_tbl$upper[i]
    
    finite_lo <- is.finite(lo)
    finite_hi <- is.finite(hi)
    
    ## ------------------------------------------------------------
    ## Finite interval
    ## ------------------------------------------------------------
    
    if (finite_lo && finite_hi) {
      
      if (par_tbl$transform[i] == "log") {
        par[i] <- exp(log(lo) + z * (log(hi) - log(lo)))
      } else {
        par[i] <- lo + z * (hi - lo)
      }
      
      next
    }
    
    s <- par_tbl$map_scale[i]
    
    ## ------------------------------------------------------------
    ## [lower, +Inf)
    ## ------------------------------------------------------------
    
    if (finite_lo && !finite_hi) {
      z_safe <- min(z, 1 - eps)
      par[i] <- lo + s * z_safe / (1 - z_safe)
      
      next
    }
    
    ## ------------------------------------------------------------
    ## (-Inf, upper]
    ## ------------------------------------------------------------
    
    if (!finite_lo && finite_hi) {
      z_safe <- max(z, eps)
      par[i] <- hi - s * (1 - z_safe) / z_safe
      
      next
    }
    
    ## ------------------------------------------------------------
    ## (-Inf, +Inf)
    ## ------------------------------------------------------------
    
    z_safe <- min(max(z, eps), 1 - eps)
    center <- par_tbl$map_center[i]
    par[i] <- center + s * tan(pi * (z_safe - 0.5))
    
  }
  
  names(par) <- par_tbl$name
  
  par
}

par_vector_to_HG_vectors <- function(
    par,
    par_tbl,
    fit_spec
  ) {

  par <- as.numeric(par)

  if (length(par) != nrow(par_tbl)) {
    stop("Parameter vector has incorrect length.")
  }

  out <- list(
    logB_vec     = as.numeric(fit_spec$logB_vec$start),
    response_vec = as.numeric(fit_spec$response_vec$start)
  )

  for (i in seq_len(nrow(par_tbl))) {

    group <- par_tbl$group[i]
    index <- par_tbl$index[i]

    out[[group]][index] <- par[i]
  }

  out
}

## Single simulation prediction ------------------------------------------------

simulate_fit_candidate <- function(
    par,
    par_tbl,
    fit_spec,
    fixed_params,
    verbose = TRUE
) {
  
  fitted_vectors <- par_vector_to_HG_vectors(
    par      = par,
    par_tbl  = par_tbl,
    fit_spec = fit_spec
  )
  
  ## fitted vectors deliberately override anything of the same name
  ## accidentally present in fixed_params.
  sim_args <- modifyList(
    fixed_params,
    fitted_vectors
  )
  
  wrapper_out <- tryCatch(
    
    do.call(
      simulate_HG_titration,
      sim_args
    ),
    
    error = function(e) {
      
      if (verbose) {
        message(
          "Simulation failed: ",
          conditionMessage(e)
        )
      }
      
      NULL
    }
  )
  
  if (
    is.null(wrapper_out)                       ||
    !is.list(wrapper_out)                      ||
    is.null(wrapper_out$tps_result)            ||
    is.null(wrapper_out$tps_result$prediction) ||
    is.null(wrapper_out$tps_result$prediction$signal_pred)
  ) {
    return(NULL)
  }
  
  pred <- as.numeric(
    wrapper_out$tps_result$prediction$signal_pred
  )
  
  if (any(!is.finite(pred))) {
    return(NULL)
  }
  
  list(
    pred       = pred,
    sim        = wrapper_out$sim,
    tps_result = wrapper_out$tps_result,
    wrapper    = wrapper_out,
    sim_args   = sim_args
  )
}

## Objective factory -----------------------------------------------------------

make_HG_fit_objective <- function(
    fit_data,
    fit_spec,
    fixed_params,
    signal_col   = "signal",
    weights      = NULL,
    loss         = c("sse", "mae", "huber"),
    huber_delta  = 1,
    penalty_fail = 1e100,
    verbose      = TRUE
) {
  
  loss <- match.arg(loss)
  
  if (!is.data.frame(fit_data)) {
    stop("`fit_data` must be a data.frame.")
  }
  
  if (!signal_col %in% names(fit_data)) {
    stop(
      sprintf(
        "Column `%s` not found in fit_data.",
        signal_col
      )
    )
  }
  
  y_obs <- as.numeric(
    fit_data[[signal_col]]
  )
  
  if (
    length(y_obs) < 2L ||
    any(!is.finite(y_obs))
  ) {
    stop("Observed signal must contain at least two finite values.")
  }
  
  ## --------------------------------------------------------------
  ## Optional weights
  ## --------------------------------------------------------------
  
  if (is.null(weights)) {
    
    weights <- rep(
      1,
      length(y_obs)
    )
    
  } else if (
    is.character(weights) &&
    length(weights) == 1L
  ) {
    
    if (!weights %in% names(fit_data)) {
      stop("Weight column not found in fit_data.")
    }
    
    weights <- fit_data[[weights]]
  }
  
  weights <- as.numeric(weights)
  
  if (
    length(weights) != length(y_obs) ||
    any(!is.finite(weights))         ||
    any(weights < 0)
  ) {
    stop(
      "`weights` must contain one finite nonnegative value per observation."
    )
  }
  
  ## --------------------------------------------------------------
  ## Parameter specification
  ## --------------------------------------------------------------
  
  if (is.null(fixed_params$species_vec)) {
    stop("`fixed_params$species_vec` is required.")
  }
  
  prep <- prepare_HG_fit_spec(
    fit_spec    = fit_spec,
    species_vec = fixed_params$species_vec
  )
  
  par_tbl             <- prep$par_table
  fit_spec_normalized <- prep$fit_spec
  
  ## --------------------------------------------------------------
  ## Loss
  ## --------------------------------------------------------------
  
  compute_loss <- function(res) {
    
    val <- switch(
      loss,
      sse = sum(res^2),
      mae = sum(abs(res)),
      huber = {
        a <- abs(res)
        sum(
          ifelse(
            a <= huber_delta,
            0.5 * res^2,
            huber_delta * (a - 0.5 * huber_delta)
          )
        )
      }
    )
    
    if (!is.finite(val)) {
      penalty_fail
    } else {
      val
    }
  }
  
  ## --------------------------------------------------------------
  ## Simulator
  ## --------------------------------------------------------------
  
  run_sim <- function(par) {
    
    simulate_fit_candidate(
      par          = par,
      par_tbl      = par_tbl,
      fit_spec     = fit_spec_normalized,
      fixed_params = fixed_params,
      verbose      = verbose
    )
  }
  
  sim_ok <- function(x) {
    
    !is.null(x)                      &&
     is.list(x)                      &&
    !is.null(x$pred)                 &&
     length(x$pred) == length(y_obs) &&
     all(is.finite(x$pred))
  }
  
  ## --------------------------------------------------------------
  ## Residuals
  ## --------------------------------------------------------------
  
  residuals <- function(par) {
    
    pred_obj <- run_sim(par)
    
    if (!sim_ok(pred_obj)) {
      
      return(
        rep(
          sqrt(penalty_fail / length(y_obs)),
          length(y_obs)
        )
      )
    }
    
    res <- (pred_obj$pred - y_obs) * sqrt(weights)
    
    res[!is.finite(res)] <- sqrt(penalty_fail / length(y_obs))
    
    res
  }
  
  ## --------------------------------------------------------------
  ## Scalar objective
  ## --------------------------------------------------------------
  
  objective <- function(par) {
    
    pred_obj <- run_sim(par)
    
    if (!sim_ok(pred_obj)) {
      return(penalty_fail)
    }
    
    res <- (pred_obj$pred - y_obs) * sqrt(weights)
    
    if (any(!is.finite(res))) {
      return(penalty_fail)
    }
    
    compute_loss(res)
  }
  
  list(
    objective    = objective,
    residuals    = residuals,
    fit_data     = fit_data,
    y_obs        = y_obs,
    weights      = weights,
    par_tbl      = par_tbl,
    fit_spec     = fit_spec_normalized,
    fixed_params = fixed_params,
    loss         = loss
  )
}

## Main fitting wrapper --------------------------------------------------------

fit_HG_titration <- function(
    fit_data,
    fixed_params,
    fit_spec,
    signal_col = "signal",
    weights    = NULL,
    method = c(
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
    verbose         = TRUE,
    checkpoint_path = NULL
) {
  
  method <- match.arg(method)
  loss   <- match.arg(loss)
  nloptr_algorithm <- match.arg(nloptr_algorithm)
  
  if (
    method == "minpack.lm" &&
    loss   != "sse"
  ) {
    stop(
      "method = 'minpack.lm' requires loss = 'sse'."
    )
  }
  
  obj <- make_HG_fit_objective(
    fit_data     = fit_data,
    fit_spec     = fit_spec,
    fixed_params = fixed_params,
    signal_col   = signal_col,
    weights      = weights,
    loss         = loss,
    huber_delta  = huber_delta,
    penalty_fail = penalty_fail,
    verbose      = verbose
  )
  
  par_tbl <- obj$par_tbl
  
  start        <- par_tbl$start
  names(start) <- par_tbl$name
  
  u_start <- par_to_unit(
    start,
    par_tbl
  )
  
  u_lower <- rep(
    0,
    length(u_start)
  )
  
  u_upper <- rep(
    1,
    length(u_start)
  )
  
  objective_unit_raw <- function(u) {
    
    obj$objective(
      unit_to_par(
        u,
        par_tbl
      )
    )
  }
  
  residuals_unit_raw <- function(u) {
    
    obj$residuals(
      unit_to_par(
        u,
        par_tbl
      )
    )
  }
  
  ## --------------------------------------------------------------
  ## Checkpointing
  ## --------------------------------------------------------------
  
  if (!is.null(checkpoint_path)) {
    
    objective_unit <-
      make_checkpointed_objective(
        objective_unit         = objective_unit_raw,
        par_tbl                = par_tbl,
        checkpoint_path        = checkpoint_path,
        save_every_improvement = TRUE,
        min_save_interval_sec  = 2,
        verbose                = verbose
      )
    
  } else {
    
    objective_unit <- objective_unit_raw
  }
  
  residuals_unit <- residuals_unit_raw
  
  ## ==============================================================
  ## nloptr
  ## ==============================================================
  
  if (method == "nloptr") {
    
    if (
      !requireNamespace(
        "nloptr",
        quietly = TRUE
      )
    ) {
      stop(
        "Package `nloptr` is required."
      )
    }
    
    fit <- nloptr::nloptr(
      x0     = u_start,
      eval_f = objective_unit,
      lb     = u_lower,
      ub     = u_upper,
      opts = modifyList(
        list(
          algorithm = nloptr_algorithm
        ),
        nloptr_control
      )
    )
    
    best_u   <- fit$solution
    best_val <- fit$objective
  }
  
  ## ==============================================================
  ## minpack.lm
  ## ==============================================================
  
  if (method == "minpack.lm") {
    
    if (
      !requireNamespace(
        "minpack.lm",
        quietly = TRUE
      )
    ) {
      stop(
        "Package `minpack.lm` is required."
      )
    }
    
    ctrl <- do.call(
      minpack.lm::nls.lm.control,
      minpack_control
    )
    
    fit <- minpack.lm::nls.lm(
      par     = u_start,
      fn      = residuals_unit,
      lower   = u_lower,
      upper   = u_upper,
      control = ctrl
    )
    
    best_u <- fit$par
    
    best_val <- sum(
      obj$residuals(
        unit_to_par(
          best_u,
          par_tbl
        )
      )^2
    )
  }
  
  ## --------------------------------------------------------------
  ## Physical parameters
  ## --------------------------------------------------------------
  
  best_par <- unit_to_par(
    best_u,
    par_tbl
  )
  
  names(best_par) <- par_tbl$name
  
  ## --------------------------------------------------------------
  ## Final prediction
  ## --------------------------------------------------------------
  
  pred_obj <- simulate_fit_candidate(
    par          = best_par,
    par_tbl      = par_tbl,
    fit_spec     = obj$fit_spec,
    fixed_params = fixed_params,
    verbose      = verbose
  )
  
  if (
    is.null(pred_obj)     ||
    length(pred_obj$pred) !=
    nrow(fit_data)
  ) {
    stop(
      "Final simulation failed at the fitted parameter vector."
    )
  }
  
  prediction <- fit_data
  
  prediction$signal_pred <- pred_obj$pred
  
  prediction$residual <- pred_obj$pred - obj$y_obs
  
  fitted_vectors <- par_vector_to_HG_vectors(
      par      = best_par,
      par_tbl  = par_tbl,
      fit_spec = obj$fit_spec
    )
  
  all_params <- modifyList(
    fixed_params,
    fitted_vectors
  )
  
  out <- list(
    method          = method,
    fit             = fit,
    objective_value = best_val,
    par             = best_par,
    par_table       = par_tbl,
    logB_vec        = fitted_vectors$logB_vec,
    response_vec    = fitted_vectors$response_vec,
    fit_spec        = obj$fit_spec,
    fixed_params    = fixed_params,
    all_params      = all_params,
    prediction      = prediction,
    sim             = pred_obj$sim,
    tps_result      = pred_obj$tps_result,
    simulation      = pred_obj$wrapper,
    loss            = loss,
    data_used       = fit_data
  )
  
  class(out) <-
    "HG_titration_fit"
  
  out
}

## Example fit -----------------------------------------------------------------

if (FALSE) { ### turn TRUE to run example (should take 40-50 iterations, approx. 1 minute)
  fixed_params <- list(
    
    species_vec = c("H1G1", "H1G2"),
    
    nsim = c(100, 500),
    
    c_stock_H = 1e-6,
    c_stock_G = 1e-1,
    V_stock_H = 3,
    
    V_G = seq(0, 0.01, length.out = 30),
    
    ## Also put any fixed TPS/grid settings here:
    max_train = 300
  )
  
  # Alternative route.
  # fixed_params <- list(
  #   species_vec      = c("H1G1", "H1G2"),
  #   nsim             = c(100, 500),
  #   conc_HG_customdf = my_concentrations
  # )
  
  fit_spec <- list(

    logB_vec = list(
      start     = c(   5,    8),
      lower     = c(-Inf, -Inf),
      upper     = c( Inf,  Inf),
      fit       = c(TRUE, TRUE),
      map_scale = c(2, 2)
    ),

    response_vec = list(
      start     = c(0, 0,  2e5,  3e5),
      lower     = c(0, 0, -Inf, -Inf),
      upper     = c(0, 0,  Inf,  Inf),
      fit       = c(FALSE, FALSE, TRUE, TRUE),
      map_scale = c(NA, NA, 2e5, 3e5)
    )
  )
  
  set.seed(0)
  
  fit_data <- sim1$tps_result$prediction |> select(total_H, total_G, signal_pred) |> 
    mutate(signal = signal_pred + rnorm(n = n(), mean = 0, sd = 1e-3))
  
  fit1 <- fit_HG_titration(
    fit_data     = fit_data,
    fixed_params = fixed_params,
    fit_spec     = fit_spec,
    signal_col   = "signal",
    
    method           = "nloptr",
    nloptr_algorithm = "NLOPT_LN_BOBYQA",
    loss             = "sse",
    
    nloptr_control = list(
      maxeval     = 100,
      xtol_rel    = 1e-5,
      print_level = 1
    ),
    
    verbose = TRUE
  )
  
  fit1$logB_vec
  fit1$response_vec
  
  ggplot(data    = fit1$prediction,
         mapping = aes(x = total_G)) +
    geom_point(
      mapping = aes(y = signal),
      shape   = 1
    ) +
    geom_line(
      mapping = aes(y = signal_pred),
      color   = "red"
    ) +
    theme_bw()
}
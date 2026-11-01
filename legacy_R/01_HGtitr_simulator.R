################################################################################
## Simulation of Host-Guest titration data
################################################################################

## Loading libraries -----------------------------------------------------------

library(dplyr)
library(ggplot2)
library(fields)

## Helper functions for simulation ---------------------------------------------

## --------------------------------------------------------------
## Parser for species names and stoichiometry.
## --------------------------------------------------------------
parse_species <- function(x) {
  
  if (
    !is.character(x) ||
    length(x) < 1L
  ) {
    stop("Species names must be a nonempty character vector.")
  }
  
  ## Examples accepted:
  ## H1G1
  ## H1G1a
  ## H1G1b
  ## H1G1state1
  ## H2G3conf_A
  pattern <-
    "^H([0-9]+)G([0-9]+)([A-Za-z][A-Za-z0-9_]*)?$"
  
  valid <- grepl(
    pattern,
    x
  )
  
  if (!all(valid)) {
    stop(
      "Invalid species name(s): ",
      paste(
        x[!valid],
        collapse = ", "
      ),
      ". Species must be written as H<number>G<number> ",
      "with an optional form suffix, for example ",
      "H1G1, H1G1a, H1G1b, H1G2state1."
    )
  }
  
  matches <- regmatches(
    x,
    regexec(pattern, x)
  )
  
  nH <- as.integer(
    vapply(
      matches,
      function(z) z[[2L]],
      character(1L)
    )
  )
  
  nG <- as.integer(
    vapply(
      matches,
      function(z) z[[3L]],
      character(1L)
    )
  )
  
  form <- vapply(
    matches,
    function(z) {
      if (
        length(z) >= 4L &&
        nzchar(z[[4L]])
      ) {
        z[[4L]]
      } else {
        ""
      }
    },
    character(1L)
  )
  
  stoichiometry <- paste0(
    "H",
    nH,
    "G",
    nG
  )
  
  ## Canonicalized full species label.
  species <- paste0(
    stoichiometry,
    form
  )
  
  data.frame(
    species          = species,
    stoichiometry    = stoichiometry,
    form             = form,
    nH               = nH,
    nG               = nG,
    stringsAsFactors = FALSE
  )
}

## --------------------------------------------------------------
## Parser for total concentration ranges.
## Interpret the supplied concentrations as TOTAL concentrations
##
## Length 2: lower and upper limits.
## Other length: exact experimental values, whose range defines the
##               simulated total-concentration domain.
## --------------------------------------------------------------
parse_total_input <- function(x, label) {
  
  if (
    !is.numeric(x)     ||
    length(x) < 1L     ||
    any(!is.finite(x)) ||
    any(x < 0)
  ) {
    stop(
      sprintf(
        "`%s_conc_range` must contain finite, nonnegative values.",
        label
      )
    )
  }
  
  values <- sort(unique(as.numeric(x)))
  
  list(
    input  = as.numeric(x),
    values = values,
    limits = range(values),
    interpretation = if (length(x) == 2L) {
      "limits"
    } else {
      "exact_values"
    }
  )
}

## --------------------------------------------------------------
## Equilibrium calculations at specified free concentrations.
## --------------------------------------------------------------
concentration_matrix <- function(
    H_free,
    G_free,
    species_table,
    log_beta
  ) {
  
  if (length(H_free) != length(G_free)) {
    stop(
      "`H_free` and `G_free` must have equal lengths."
    )
  }
  
  if (
    any(!is.finite(H_free)) ||
    any(!is.finite(G_free)) ||
    any(H_free < 0)         ||
    any(G_free < 0)
  ) {
    stop(
      "Free concentrations must be finite and nonnegative."
    )
  }
  
  n_point   <- length(H_free)
  n_species <- length(species_table$species)
  
  logH <- rep(-Inf, n_point)
  logG <- rep(-Inf, n_point)
  
  positive_H <- H_free > 0
  positive_G <- G_free > 0
  
  logH[positive_H] <- log(H_free[positive_H])
  logG[positive_G] <- log(G_free[positive_G])
  
  out <- matrix(
    0,
    nrow = n_point,
    ncol = n_species
  )
  
  log_double_max <- log(.Machine$double.xmax)
  
  for (j in seq_len(n_species)) {
    
    log_conc <- rep(log_beta[j], n_point)
    
    if (species_table$nH[j] > 0L) {
      log_conc <- log_conc +
        species_table$nH[j] * logH
    }
    
    if (species_table$nG[j] > 0L) {
      log_conc <- log_conc +
        species_table$nG[j] * logG
    }
    
    if (any(log_conc > log_double_max)) {
      stop(
        "A species concentration overflowed numerically. ",
        "The equilibrium constants or concentration ranges ",
        "are too large for double-precision arithmetic."
      )
    }
    
    out[, j] <- exp(log_conc)
  }
  
  colnames(out) <- species_table$species
  
  out
}

calculate_state <- function(
    H_free,
    G_free,
    species,
    log_beta
  ) {
  
  conc <- concentration_matrix(
    H_free        = H_free,
    G_free        = G_free,
    species_table = species,
    log_beta      = log_beta
  )
  
  total_H <- drop(
    conc %*% species$nH
  )
  
  total_G <- drop(
    conc %*% species$nG
  )
  
  signal <- drop(
    conc %*% species$response
  )
  
  list(
    concentrations = conc,
    total_H        = total_H,
    total_G        = total_G,
    signal         = signal
  )
}

## --------------------------------------------------------------
## Adaptively determine free H and G for one total-H/total-G boundary point.
##
## The multiplicative update is performed in log space:
##
## log(H_free,new) =
##   log(H_free,old) +
##   damping * log(H_target / H_calculated)
##
## and analogously for G.
## --------------------------------------------------------------
calibrate_corner <- function(
    target_H,
    target_G,
    species,
    log_beta,
    endpoint_tol       = 1e-7,
    endpoint_maxiter   = 500L,
    endpoint_damping   = 0.5,
    use_optim_fallback = TRUE
  ) {
  
  log_minimum <- log(.Machine$double.xmin)
  
  log_free_H <- if (target_H > 0) {
    log(target_H)
  } else {
    -Inf
  }
  
  log_free_G <- if (target_G > 0) {
    log(target_G)
  } else {
    -Inf
  }
  
  calculate_residuals <- function(
      free_H,
      free_G,
      species,
      log_beta
    ) {
    
    state <- calculate_state(
      H_free   = free_H,
      G_free   = free_G,
      species  = species,
      log_beta = log_beta
    )
    
    residual_H <- if (target_H > 0) {
      log(state$total_H / target_H)
    } else if (state$total_H == 0) {
      0
    } else {
      Inf
    }
    
    residual_G <- if (target_G > 0) {
      log(state$total_G / target_G)
    } else if (state$total_G == 0) {
      0
    } else {
      Inf
    }
    
    list(
      state         = state,
      residual_H    = residual_H,
      residual_G    = residual_G,
      maximum_error = max(
        abs(c(residual_H, residual_G))
      )
    )
  }
  
  converged   <- FALSE
  method_used <- "multiplicative"
  iteration   <- 0L
  
  for (
    iteration in seq.int(
      from = 0L,
      to   = endpoint_maxiter
    )
  ) {
    
    free_H <- if (target_H > 0) {
      exp(log_free_H)
    } else {
      0
    }
    
    free_G <- if (target_G > 0) {
      exp(log_free_G)
    } else {
      0
    }
    
    evaluation <- calculate_residuals(
      free_H   = free_H,
      free_G   = free_G,
      species  = species,
      log_beta = log_beta
    )
    
    if (
      is.finite(evaluation$maximum_error) &&
      evaluation$maximum_error <= endpoint_tol
    ) {
      converged <- TRUE
      break
    }
    
    if (iteration == endpoint_maxiter) {
      break
    }
    
    if (target_H > 0) {
      
      log_free_H <- log_free_H -
        endpoint_damping * evaluation$residual_H
      
      ## Since total H always contains free H,
      ## free H cannot physically exceed total H.
      log_free_H <- min(
        log_free_H,
        log(target_H)
      )
      
      log_free_H <- max(
        log_free_H,
        log_minimum
      )
    }
    
    if (target_G > 0) {
      
      log_free_G <- log_free_G -
        endpoint_damping * evaluation$residual_G
      
      ## Since total G always contains free G,
      ## free G cannot physically exceed total G.
      log_free_G <- min(
        log_free_G,
        log(target_G)
      )
      
      log_free_G <- max(
        log_free_G,
        log_minimum
      )
    }
  }
  
  ## --------------------------------------------------------------
  ## Optional base-R optimizer fallback
  ##
  ## This is used only if the inexpensive multiplicative iteration
  ## stalls. It is still applied only to a boundary corner, not to
  ## every experimental measurement.
  ## --------------------------------------------------------------
  
  if (!converged && isTRUE(use_optim_fallback)) {
    
    targets <- c(
      H = target_H,
      G = target_G
    )
    
    active <- targets > 0
    
    initial_logs <- c(
      H = log_free_H,
      G = log_free_G
    )[active]
    
    lower_logs <- pmax(
      log_minimum,
      log(targets[active]) - 700
    )
    
    upper_logs <- log(targets[active])
    
    objective <- function(active_logs) {
      
      full_logs <- c(
        H = -Inf,
        G = -Inf
      )
      
      full_logs[active] <- active_logs
      
      trial_H <- if (target_H > 0) {
        exp(full_logs["H"])
      } else {
        0
      }
      
      trial_G <- if (target_G > 0) {
        exp(full_logs["G"])
      } else {
        0
      }
      
      trial <- calculate_residuals(
        free_H   = trial_H,
        free_G   = trial_G,
        species  = species,
        log_beta = log_beta
      )
      
      residuals <- c(
        trial$residual_H,
        trial$residual_G
      )
      
      if (any(!is.finite(residuals))) {
        return(.Machine$double.xmax / 100)
      }
      
      sum(residuals^2)
    }
    
    optimization <- stats::optim(
      par     = initial_logs,
      fn      = objective,
      method  = "L-BFGS-B",
      lower   = lower_logs,
      upper   = upper_logs,
      control = list(
        maxit = endpoint_maxiter
      )
    )
    
    optimized_logs <- c(
      H = -Inf,
      G = -Inf
    )
    
    optimized_logs[active] <- optimization$par
    
    log_free_H <- optimized_logs["H"]
    log_free_G <- optimized_logs["G"]
    
    free_H <- if (target_H > 0) {
      exp(log_free_H)
    } else {
      0
    }
    
    free_G <- if (target_G > 0) {
      exp(log_free_G)
    } else {
      0
    }
    
    evaluation <- calculate_residuals(
      free_H   = free_H,
      free_G   = free_G,
      species  = species,
      log_beta = log_beta
    )
    
    converged <-
      is.finite(evaluation$maximum_error) &&
      evaluation$maximum_error <= endpoint_tol
    
    method_used <- "optim_fallback"
  }
  
  if (!converged) {
    stop(
      sprintf(
        paste0(
          "Could not calibrate the free-concentration boundary ",
          "for total_H = %.6g and total_G = %.6g. ",
          "Final maximum log-relative error: %.6g."
        ),
        target_H,
        target_G,
        evaluation$maximum_error
      )
    )
  }
  
  free_H <- if (target_H > 0) {
    exp(log_free_H)
  } else {
    0
  }
  
  free_G <- if (target_G > 0) {
    exp(log_free_G)
  } else {
    0
  }
  
  final_state <- calculate_state(
    H_free   = free_H,
    G_free   = free_G,
    species  = species,
    log_beta = log_beta
  )
  
  data.frame(
    target_total_H   = target_H,
    target_total_G   = target_G,
    free_H           = free_H,
    free_G           = free_G,
    achieved_total_H = final_state$total_H,
    achieved_total_G = final_state$total_G,
    log_error_H = if (target_H > 0) {
      log(final_state$total_H / target_H)
    } else {
      0
    },
    log_error_G = if (target_G > 0) {
      log(final_state$total_G / target_G)
    } else {
      0
    },
    iterations = iteration,
    method     = method_used,
    converged  = converged,
    stringsAsFactors = FALSE
  )
}

## --------------------------------------------------------------
## Create logarithmic axis grid.
## --------------------------------------------------------------
make_log_axis <- function(
    free_limits,
    corner_values,
    label,
    nlength,
    zero_floor_decades = 8
) {
  
  lower <- free_limits[1L]
  upper <- free_limits[2L]
  
  if (upper < 0 || lower < 0) {
    stop(
      sprintf(
        "Negative calibrated free-%s concentration encountered.",
        label
      )
    )
  }
  
  ## Entire axis is zero.
  if (upper == 0) {
    return(0)
  }
  
  ## Degenerate positive axis.
  if (lower == upper) {
    return(lower)
  }
  
  ## Ordinary positive logarithmic axis.
  if (lower > 0) {
    return(
      exp(
        seq(
          log(lower),
          log(upper),
          length.out = nlength
        )
      )
    )
  }
  
  ## Axis begins at exactly zero.
  if (nlength == 2L) {
    return(c(0, upper))
  }
  
  positive_corners <- corner_values[
    corner_values > 0
  ]
  
  numerical_floor <- upper *
    10^(-zero_floor_decades)
  
  positive_lower <- min(
    c(
      positive_corners,
      numerical_floor
    )
  )
  
  positive_lower <- max(
    positive_lower,
    .Machine$double.xmin
  )
  
  positive_axis <- exp(
    seq(
      log(positive_lower),
      log(upper),
      length.out = nlength - 1L
    )
  )
  
  unique(
    c(
      0,
      positive_axis
    )
  )
}

## --------------------------------------------------------------
## Apply padded limits.
## --------------------------------------------------------------
padded_limits <- function(
    x,
    total_padding_decades = 0.05
  ) {
  c(
    lower = if (x[1L] == 0) {
      0
    } else {
      x[1L] * 10^(-total_padding_decades)
    },
    upper = x[2L] * 10^(total_padding_decades)
  )
}

## --------------------------------------------------------------
## Binary search on grid to find relevant slice.
## --------------------------------------------------------------
find_relevant_grid <- function(
    H_axis,
    G_axis,
    H_limits,
    G_limits,
    species,
    log_beta,
    calculate_state
) {
  
  nH <- length(H_axis)
  nG <- length(G_axis)
  
  result <- vector("list", nH)
  
  n_boundary_evaluations <- 0L
  
  for (i in seq_len(nH)) {
    
    H_free_i <- H_axis[i]
    
    ## Cache equilibrium calculations during the binary searches
    ## for this particular free-H value.
    cache <- new.env(
      hash   = TRUE,
      parent = emptyenv()
    )
    
    evaluate_G <- function(j) {
      
      key <- as.character(j)
      
      if (!exists(
        key,
        envir    = cache,
        inherits = FALSE
      )) {
        
        state <- calculate_state(
          H_free   = H_free_i,
          G_free   = G_axis[j],
          species  = species,
          log_beta = log_beta
        )
        
        value <- c(
          total_H = state$total_H,
          total_G = state$total_G
        )
        
        assign(
          key,
          value,
          envir = cache
        )
        
        n_boundary_evaluations <<-
          n_boundary_evaluations + 1L
      }
      
      get(
        key,
        envir    = cache,
        inherits = FALSE
      )
    }
    
    ## ------------------------------------------------------------
    ## Binary-search helpers
    ## ------------------------------------------------------------
    
    first_ge <- function(component, target) {
      
      first_value <- evaluate_G(1L)[component]
      
      if (first_value >= target) {
        return(1L)
      }
      
      last_value <- evaluate_G(nG)[component]
      
      if (last_value < target) {
        return(nG + 1L)
      }
      
      lo <- 1L
      hi <- nG
      
      while (lo < hi) {
        
        mid <- floor((lo + hi) / 2)
        
        value <- evaluate_G(mid)[component]
        
        if (value >= target) {
          hi <- mid
        } else {
          lo <- mid + 1L
        }
      }
      
      lo
    }
    
    last_le <- function(component, target) {
      
      first_value <- evaluate_G(1L)[component]
      
      if (first_value > target) {
        return(0L)
      }
      
      last_value <- evaluate_G(nG)[component]
      
      if (last_value <= target) {
        return(nG)
      }
      
      lo <- 1L
      hi <- nG
      
      while (lo < hi) {
        
        mid <- ceiling((lo + hi) / 2)
        
        value <- evaluate_G(mid)[component]
        
        if (value <= target) {
          lo <- mid
        } else {
          hi <- mid - 1L
        }
      }
      
      lo
    }
    
    ## ------------------------------------------------------------
    ## Quick rejection of entire free-G row
    ## ------------------------------------------------------------
    
    state_low  <- evaluate_G(1L)
    state_high <- evaluate_G(nG)
    
    ## Even the smallest G is already above the domain.
    if (
      state_low["total_H"] > H_limits["upper"] ||
      state_low["total_G"] > G_limits["upper"]
    ) {
      next
    }
    
    ## Even the largest G is still below the domain.
    if (
      state_high["total_H"] < H_limits["lower"] ||
      state_high["total_G"] < G_limits["lower"]
    ) {
      next
    }
    
    ## ------------------------------------------------------------
    ## Intersection of the H-total and G-total admissible intervals
    ## ------------------------------------------------------------
    
    j_lower <- max(
      first_ge(
        "total_H",
        H_limits["lower"]
      ),
      first_ge(
        "total_G",
        G_limits["lower"]
      )
    )
    
    j_upper <- min(
      last_le(
        "total_H",
        H_limits["upper"]
      ),
      last_le(
        "total_G",
        G_limits["upper"]
      )
    )
    
    if (j_lower <= j_upper) {
      
      ind <- seq.int(
        j_lower,
        j_upper
      )
      
      result[[i]] <- data.frame(
        free_H = rep(
          H_free_i,
          length(ind)
        ),
        free_G     = G_axis[ind],
        point_type = "grid",
        stringsAsFactors = FALSE
      )
    }
  }
  
  result <- Filter(
    Negate(is.null),
    result
  )
  
  if (length(result) == 0L) {
    stop(
      "No free-concentration grid points fall inside the ",
      "padded requested total-concentration domain."
    )
  }
  
  grid_points <- do.call(
    rbind,
    result
  )
  
  rownames(grid_points) <- NULL
  
  attr(
    grid_points,
    "n_boundary_evaluations"
  ) <- n_boundary_evaluations
  
  grid_points
}

## Main simulator of species concentrations and signal -------------------------

HG_titr_matrix <- function(
    species_vec,
    logB_vec,
    response_vec,
    H_conc_range,
    G_conc_range,
    nsim,
    endpoint_tol          = 1e-7,
    endpoint_maxiter      = 500L,
    endpoint_damping      = 0.5,
    use_optim_fallback    = TRUE,
    include_corner_points = TRUE,
    zero_floor_decades    = 8,
    trim_total_domain     = TRUE,
    total_padding_decades = 0.05
) {
  
  ## ================================================================
  ## Input validation
  ## ================================================================
  
  if (!is.character(species_vec)) {
    stop("`species_vec` must be a character vector.")
  }
  
  if (
    !is.numeric(logB_vec) ||
    any(!is.finite(logB_vec))
  ) {
    stop("`logB_vec` must contain finite numeric values.")
  }
  
  if (
    !is.numeric(response_vec) ||
    any(!is.finite(response_vec))
  ) {
    stop("`response_vec` must contain finite numeric values.")
  }
  
  if (length(species_vec) != length(logB_vec)) {
    stop("`species_vec` and `logB_vec` must have the same length.")
  }
  
  if (length(response_vec) != length(species_vec) + 2L) {
    stop(
      "`response_vec` must contain responses in the order ",
      "H1G0, H0G1, followed by the species in `species_vec`."
    )
  }
  
  if (
    length(nsim) > 2L     ||
    !is.numeric(nsim)     ||
    any(!is.finite(nsim)) ||
    any(nsim < 20)        ||
    any(nsim != as.integer(nsim))
  ) {
    stop("`nsim` must be 1- or 2-length integer vector with at least 20 simulations.")
  }
  
  nsim <- as.integer(nsim)
  
  if (
    length(endpoint_tol) != 1L ||
    !is.finite(endpoint_tol)   ||
    endpoint_tol <= 0
  ) {
    stop("`endpoint_tol` must be a positive finite number.")
  }
  
  if (
    length(endpoint_maxiter) != 1L ||
    endpoint_maxiter < 1           ||
    endpoint_maxiter != as.integer(endpoint_maxiter)
  ) {
    stop("`endpoint_maxiter` must be a positive integer.")
  }
  
  endpoint_maxiter <- as.integer(endpoint_maxiter)
  
  if (
    length(endpoint_damping) != 1L ||
    !is.finite(endpoint_damping)   ||
    endpoint_damping <= 0          ||
    endpoint_damping > 1
  ) {
    stop("`endpoint_damping` must lie in (0, 1].")
  }
  
  if (
    length(zero_floor_decades) != 1L ||
    !is.finite(zero_floor_decades)   ||
    zero_floor_decades <= 0          ||
    zero_floor_decades > 300
  ) {
    stop("`zero_floor_decades` must be between 0 and 300.")
  }
  
  ## ================================================================
  ## Parse species names
  ## ================================================================
  
  parsed_species <- parse_species(species_vec)
  
  if (any(parsed_species$nH + parsed_species$nG == 0L)) {
    stop("The null species H0G0 is not permitted.")
  }
  
  reserved_free_species <-
    (
      parsed_species$nH == 1L &
        parsed_species$nG == 0L
    ) |
    (
      parsed_species$nH == 0L &
        parsed_species$nG == 1L
    )
  
  if (any(reserved_free_species)) {
    
    stop(
      "H1G0 and H0G1 stoichiometries are reserved for the ",
      "automatically added free H and free G species."
    )
  }
  
  if (anyDuplicated(parsed_species$species)) {
    
    duplicated_species <- unique(
      parsed_species$species[
        duplicated(parsed_species$species)
      ]
    )
    
    stop(
      "Duplicated species after parsing: ",
      paste(duplicated_species, collapse = ", ")
    )
  }
  
  ## ================================================================
  ## Complete species table and concentration ranges
  ## ================================================================
  
  species_table <- data.frame(
    species = c(
      "H1G0",
      "H0G1",
      parsed_species$species
    ),
    nH = c(
      1L,
      0L,
      parsed_species$nH
    ),
    nG = c(
      0L,
      1L,
      parsed_species$nG
    ),
    logB = c(
      0,
      0,
      logB_vec
    ),
    response = response_vec,
    stringsAsFactors = FALSE
  )
  
  ## Natural-log cumulative constants, retained in log space.
  log_beta <- log(10) * species_table$logB
  
  H_total_spec <- parse_total_input(
    H_conc_range,
    "H"
  )
  
  G_total_spec <- parse_total_input(
    G_conc_range,
    "G"
  )
  
  ## ================================================================
  ## Calibrate all unique corners of the requested total domain
  ## ================================================================
  
  target_corners <- unique(
    expand.grid(
      total_H = H_total_spec$limits,
      total_G = G_total_spec$limits,
      KEEP.OUT.ATTRS   = FALSE,
      stringsAsFactors = FALSE
    )
  )
  
  corner_results <- lapply(
    seq_len(nrow(target_corners)),
    function(i) {
      calibrate_corner(
        target_H           = target_corners$total_H[i],
        target_G           = target_corners$total_G[i],
        species            = species_table,
        log_beta           = log_beta,
        endpoint_tol       = endpoint_tol,
        endpoint_maxiter   = endpoint_maxiter,
        endpoint_damping   = endpoint_damping,
        use_optim_fallback = use_optim_fallback
      )
    }
  )
  
  corner_diagnostics <- do.call(
    rbind,
    corner_results
  )
  
  rownames(corner_diagnostics) <- NULL
  
  corner_diagnostics$corner_id <- paste0(
    "corner_",
    seq_len(nrow(corner_diagnostics))
  )
  
  corner_diagnostics <- corner_diagnostics[
    ,
    c(
      "corner_id",
      "target_total_H",
      "target_total_G",
      "free_H",
      "free_G",
      "achieved_total_H",
      "achieved_total_G",
      "log_error_H",
      "log_error_G",
      "iterations",
      "method",
      "converged"
    )
  ]
  
  ## The free-axis limits are the envelopes of the free
  ## concentrations required at every total-concentration corner.
  
  H_free_limits <- range(
    corner_diagnostics$free_H
  )
  
  G_free_limits <- range(
    corner_diagnostics$free_G
  )
  
  ## ================================================================
  ## Construct logarithmic free-concentration axes
  ##
  ## If an axis includes zero, zero is included explicitly and the
  ## remaining nsim - 1 points form a logarithmic positive grid.
  ## ================================================================
  
  if (length(nsim) == 1L) {
    nsim <- rep(nsim, 2L)
  }
  
  nsim_H <- as.integer(nsim[1L])
  nsim_G <- as.integer(nsim[2L])
  
  H_axis <- make_log_axis(
    free_limits        = H_free_limits,
    corner_values      = corner_diagnostics$free_H,
    label              = "H",
    nlength            = nsim_H,
    zero_floor_decades = zero_floor_decades
  )
  
  G_axis <- make_log_axis(
    free_limits        = G_free_limits,
    corner_values      = corner_diagnostics$free_G,
    label              = "G",
    nlength            = nsim_G,
    zero_floor_decades = zero_floor_decades
  )
  
  ## ================================================================
  ## Cartesian product of free H and G narrowed by binary search
  ## ================================================================
  
  H_keep <- padded_limits(
    H_total_spec$limits,
    total_padding_decades = total_padding_decades
  )
  
  G_keep <- padded_limits(
    G_total_spec$limits,
    total_padding_decades = total_padding_decades
  )
  
  grid_points <- find_relevant_grid(
    H_axis   = H_axis,
    G_axis   = G_axis,
    H_limits = H_keep,
    G_limits = G_keep,
    species  = species_table,
    log_beta = log_beta,
    calculate_state = calculate_state
  )
  
  ## ================================================================
  ## Calculate all species, totals, and signal
  ## ================================================================
  
  final_state <- calculate_state(
    H_free   = grid_points$free_H,
    G_free   = grid_points$free_G,
    species  = species_table,
    log_beta = log_beta
  )
  
  concentration_data <- as.data.frame(
    final_state$concentrations,
    check.names = FALSE
  )
  
  names(concentration_data) <- paste0(
    "conc_",
    names(concentration_data)
  )
  
  surface_data <- cbind(
    data.frame(
      point_type = grid_points$point_type,
      free_H     = grid_points$free_H,
      free_G     = grid_points$free_G,
      total_H    = final_state$total_H,
      total_G    = final_state$total_G,
      signal     = final_state$signal,
      stringsAsFactors = FALSE
    ),
    concentration_data
  )
  
  # Apply trimming.
  keep <- with(
    surface_data,
    total_H >= H_keep["lower"] &
    total_H <= H_keep["upper"] &
    total_G >= G_keep["lower"] &
    total_G <= G_keep["upper"]
  )
  
  surface_data <- surface_data[
    keep,
    ,
    drop = FALSE
  ]
  
  rownames(surface_data) <- NULL
  
  ## Add the calibrated corner points explicitly.
  if (isTRUE(include_corner_points)) {
    
    corner_state <- calculate_state(
      H_free   = corner_diagnostics$free_H,
      G_free   = corner_diagnostics$free_G,
      species  = species_table,
      log_beta = log_beta
    )
    
    corner_concentrations <- as.data.frame(
      corner_state$concentrations,
      check.names = FALSE
    )
    
    names(corner_concentrations) <- paste0(
      "conc_",
      names(corner_concentrations)
    )
    
    corner_surface <- cbind(
      data.frame(
        point_type = "target_corner",
        free_H     = corner_diagnostics$free_H,
        free_G     = corner_diagnostics$free_G,
        total_H    = corner_state$total_H,
        total_G    = corner_state$total_G,
        signal     = corner_state$signal,
        stringsAsFactors = FALSE
      ),
      corner_concentrations
    )
    
    surface_data <- rbind(
      surface_data,
      corner_surface
    )
    
    ## Remove any exact duplicate free-coordinate points.
    key <- paste(
      sprintf("%.17g", surface_data$free_H),
      sprintf("%.17g", surface_data$free_G),
      sep = "|"
    )
    
    surface_data <- surface_data[
      !duplicated(key),
      ,
      drop = FALSE
    ]
    
    rownames(surface_data) <- NULL
  }
  
  ## ================================================================
  ## Diagnostics for the realized total-concentration cloud
  ## ================================================================
  
  grid_diagnostics = list(
    n_full_cartesian =
      length(H_axis) * length(G_axis),
    
    n_candidate = nrow(grid_points),
    
    candidate_fraction =
      nrow(grid_points) /
      (length(H_axis) * length(G_axis)),
    
    n_boundary_evaluations =
      attr(
        grid_points,
        "n_boundary_evaluations"
      )
  )
  
  raw_min_state <- calculate_state(
    H_free   = min(H_axis),
    G_free   = min(G_axis),
    species  = species_table,
    log_beta = log_beta
  )
  
  raw_max_state <- calculate_state(
    H_free   = max(H_axis),
    G_free   = max(G_axis),
    species  = species_table,
    log_beta = log_beta
  )
  
  range_diagnostics <- data.frame(
    component = c("H", "G"),
    
    requested_min = c(
      H_total_spec$limits[1L],
      G_total_spec$limits[1L]
    ),
    
    requested_max = c(
      H_total_spec$limits[2L],
      G_total_spec$limits[2L]
    ),
    
    realized_min = c(
      min(surface_data$total_H),
      min(surface_data$total_G)
    ),
    
    realized_max = c(
      max(surface_data$total_H),
      max(surface_data$total_G)
    ),
    
    realized_min_raw = c(
      raw_min_state$total_H,
      raw_min_state$total_G
    ),
    
    realized_max_raw = c(
      raw_max_state$total_H,
      raw_max_state$total_G
    ),
    
    stringsAsFactors = FALSE
  )
  
  ## ================================================================
  ## Return
  ## ================================================================
  
  structure(
    list(
      species = species_table,
      
      requested_total = list(
        H = H_total_spec,
        G = G_total_spec
      ),
      
      calibrated_free_limits = list(
        H = H_free_limits,
        G = G_free_limits
      ),
      
      free_axes = list(
        H = H_axis,
        G = G_axis
      ),
      
      corner_diagnostics = corner_diagnostics,
      
      grid_diagnostics   = grid_diagnostics,
      
      range_diagnostics  = range_diagnostics,
      
      concentrations     = concentration_data,
      
      surface            = surface_data,
      
      grid_dimensions = c(
        H = length(H_axis),
        G = length(G_axis)
      ),
      
      nsim_requested = nsim
    ),
    class = "HG_titr_matrix"
  )
}

## Thin-plate spline fitter on titration matrix surface ------------------------

interpolate_HG_Tps <- function(
    sim,
    newdata,
    padding_decades = 0.25,
    max_train       = 1200L,
    include_corners = TRUE
) {
  
  stopifnot(
    is.list(sim),
    is.data.frame(sim$surface),
    is.data.frame(newdata),
    all(
      c("total_H", "total_G", "signal") %in%
        names(sim$surface)
    ),
    all(
      c("total_H", "total_G") %in%
        names(newdata)
    ),
    is.numeric(max_train),
    length(max_train) == 1L,
    max_train >= 10
  )
  
  max_train <- as.integer(max_train)
  
  train <- sim$surface
  
  if (
    !include_corners &&
    "point_type" %in% names(train)
  ) {
    train <- train[
      train$point_type == "grid",
      ,
      drop = FALSE
    ]
  }
  
  train <- train[
    is.finite(train$total_H)   &
      is.finite(train$total_G) &
      is.finite(train$signal)  &
      train$total_H > 0        &
      train$total_G >= 0,
    c("total_H", "total_G", "signal"),
    drop = FALSE
  ]
  
  pred <- newdata[
    is.finite(newdata$total_H)   &
      is.finite(newdata$total_G) &
      newdata$total_H > 0        &
      newdata$total_G >= 0,
    ,
    drop = FALSE
  ]
  
  if (nrow(pred) != nrow(newdata)) {
    stop(
      "`newdata` must contain positive finite `total_H` ",
      "and nonnegative finite `total_G`."
    )
  }
  
  ## --------------------------------------------------------------
  ## Common offsets and transformed coordinates
  ## --------------------------------------------------------------
  
  G_offset <- 1e-10
  
  train$x_H <- log10(train$total_H)
  train$x_G <- log10(train$total_G + G_offset)
  
  pred$x_H <- log10(pred$total_H)
  pred$x_G <- log10(pred$total_G + G_offset)
  
  ## --------------------------------------------------------------
  ## Retain a local rectangle around the prediction points
  ## --------------------------------------------------------------
  
  H_limits <- range(pred$x_H) +
    c(-padding_decades, padding_decades)
  
  G_limits <- range(pred$x_G) +
    c(-padding_decades, padding_decades)
  
  local <- train[
    train$x_H >= H_limits[1L] &
    train$x_H <= H_limits[2L] &
    train$x_G >= G_limits[1L] &
    train$x_G <= G_limits[2L],
    ,
    drop = FALSE
  ]
  
  if (nrow(local) < 20L) {
    stop(
      "Too few simulation points lie near the experimental domain. ",
      "Increase `padding_decades` or simulate a denser surface."
    )
  }
  
  ## Average exact duplicate transformed coordinates.
  local <- aggregate(
    signal ~ x_H + x_G,
    data   = local,
    FUN    = mean
  )
  
  n_local_before_thinning <- nrow(local)
  
  ## --------------------------------------------------------------
  ## Deterministic two-dimensional thinning
  ##
  ## Divide transformed concentration space into rectangular cells
  ## and retain one actual simulation point nearest each cell centre.
  ## --------------------------------------------------------------
  
  if (nrow(local) > max_train) {
    
    nbin <- max(
      2L,
      floor(sqrt(max_train))
    )
    
    H_breaks <- seq(
      min(local$x_H),
      max(local$x_H),
      length.out = nbin + 1L
    )
    
    G_breaks <- seq(
      min(local$x_G),
      max(local$x_G),
      length.out = nbin + 1L
    )
    
    bin_H <- cut(
      local$x_H,
      breaks         = H_breaks,
      include.lowest = TRUE,
      labels         = FALSE
    )
    
    bin_G <- cut(
      local$x_G,
      breaks         = G_breaks,
      include.lowest = TRUE,
      labels         = FALSE
    )
    
    group_id <- interaction(
      bin_H,
      bin_G,
      drop = TRUE
    )
    
    groups <- split(
      seq_len(nrow(local)),
      group_id
    )
    
    keep <- vapply(
      groups,
      function(index) {
        
        centre_H <- mean(local$x_H[index])
        centre_G <- mean(local$x_G[index])
        
        distance2 <-
          (local$x_H[index] - centre_H)^2 +
          (local$x_G[index] - centre_G)^2
        
        index[which.min(distance2)]
      },
      integer(1L)
    )
    
    local <- local[
      sort(unique(keep)),
      ,
      drop = FALSE
    ]
  }
  
  ## --------------------------------------------------------------
  ## Thin-plate spline and prediction
  ## --------------------------------------------------------------
  
  fit <- fields::Tps(
    x = as.matrix(
      local[, c("x_H", "x_G")]
    ),
    Y = local$signal,
    give.warnings = FALSE
  )
  
  pred$signal_pred <- as.numeric(
    predict(
      fit,
      as.matrix(
        pred[, c("x_H", "x_G")]
      )
    )
  )
  
  result             <- newdata
  result$signal_pred <- pred$signal_pred
  
  list(
    prediction    = result,
    fit           = fit,
    training_data = local,
    diagnostics   = list(
      G_offset                = G_offset,
      padding_decades         = padding_decades,
      n_surface               = nrow(train),
      n_local_before_thinning = n_local_before_thinning,
      n_training_used         = nrow(local)
    )
  )
}

## Simulator of realistic titration concentration grid -------------------------

make_HG_titration_grid <- function(
    c_stock_H,
    c_stock_G,
    V_stock_H,
    V_G,
    include_initial   = TRUE,
    V_G_is_cumulative = FALSE
) {
  
  ## --------------------------------------------------------------
  ## Input validation
  ## --------------------------------------------------------------
  
  scalar_positive <- function(x) {
    is.numeric(x)     &&
      length(x) == 1L &&
      is.finite(x)    &&
      x > 0
  }
  
  if (!scalar_positive(c_stock_H)) {
    stop("`c_stock_H` must be one positive finite number.")
  }
  
  if (!scalar_positive(c_stock_G)) {
    stop("`c_stock_G` must be one positive finite number.")
  }
  
  if (!scalar_positive(V_stock_H)) {
    stop("`V_stock_H` must be one positive finite number.")
  }
  
  if (
    !is.numeric(V_G)     ||
    length(V_G) < 1L     ||
    any(!is.finite(V_G)) ||
    any(V_G < 0)
  ) {
    stop("`V_G` must contain finite nonnegative volumes.")
  }
  
  if (
    length(include_initial) != 1L ||
    !is.logical(include_initial)  ||
    is.na(include_initial)
  ) {
    stop("`include_initial` must be TRUE or FALSE.")
  }
  
  if (
    length(V_G_is_cumulative) != 1L ||
    !is.logical(V_G_is_cumulative)  ||
    is.na(V_G_is_cumulative)
  ) {
    stop("`V_G_is_cumulative` must be TRUE or FALSE.")
  }
  
  ## --------------------------------------------------------------
  ## Cumulative guest-stock volume
  ## --------------------------------------------------------------
  
  if (V_G_is_cumulative) {
    
    if (is.unsorted(V_G, strictly = FALSE)) {
      stop(
        "When `V_G_is_cumulative = TRUE`, `V_G` must be ",
        "nondecreasing."
      )
    }
    
    V_G_cumulative <- V_G
    
    V_G_added <- c(
      V_G_cumulative[1L],
      diff(V_G_cumulative)
    )
    
  } else {
    
    V_G_added      <- V_G
    V_G_cumulative <- cumsum(V_G)
  }
  
  ## Optionally prepend the initial solution before adding guest.
  if (include_initial) {
    
    V_G_added      <- c(0, V_G_added)
    V_G_cumulative <- c(0, V_G_cumulative)
  }
  
  ## --------------------------------------------------------------
  ## Mole and volume balances
  ## --------------------------------------------------------------
  
  V_total <- V_stock_H + V_G_cumulative
  
  n_H <- c_stock_H * V_stock_H
  n_G <- c_stock_G * V_G_cumulative
  
  total_H <- n_H / V_total
  total_G <- n_G / V_total
  
  data.frame(
    titration_step = seq_along(V_G_cumulative) - as.integer(include_initial),
    V_G_added      = V_G_added,
    V_G_cumulative = V_G_cumulative,
    V_total        = V_total,
    total_H        = total_H,
    total_G        = total_G,
    stringsAsFactors = FALSE
  )
}

## Wrapper for simulating titration data ---------------------------------------

simulate_HG_titration <- function(
    ## stoichiometry & simulation parameters
    species_vec,
    logB_vec,
    response_vec,
    nsim,
    endpoint_tol          = 1e-7,
    endpoint_maxiter      = 500L,
    endpoint_damping      = 0.5,
    use_optim_fallback    = TRUE,
    include_corner_points = TRUE,
    zero_floor_decades    = 8,
    trim_total_domain     = TRUE,
    total_padding_decades = 0.05,
    
    ## titration parameters
    c_stock_H,
    c_stock_G,
    V_stock_H,
    V_G,
    include_initial   = TRUE,
    V_G_is_cumulative = FALSE,
    ## optional custom titration points (given as dataframe)
    c_HG_customdf     = NULL,
    
    ## spline interpolation parameters
    padding_decades = 0.25,
    max_train       = 1200L,
    include_corners = TRUE
  ) {
  
  if (is.null(c_HG_customdf)) {
    total_exp_conc_grid <- make_HG_titration_grid(
      c_stock_H         = c_stock_H,
      c_stock_G         = c_stock_G,
      V_stock_H         = V_stock_H,
      V_G               = V_G,
      include_initial   = include_initial,
      V_G_is_cumulative = V_G_is_cumulative
    )
  } else {
    if (
      !is.data.frame(c_HG_customdf)                                                ||
      !all(is.finite(as.matrix(c_HG_customdf)) & c_HG_customdf >= 0, na.rm = TRUE) ||
      !all(c("total_H", "total_G") %in% colnames(c_HG_customdf))
    ) {
      stop(
        "`c_HG_customdf` must contain nonnegative finite values, ",
        "with mandatory columns named total_H and total_G."
      )
    }
    
    total_exp_conc_grid <- c_HG_customdf
  }
  
  sim <- HG_titr_matrix(
    species_vec           = species_vec,
    logB_vec              = logB_vec,
    response_vec          = response_vec,
    H_conc_range          = range(total_exp_conc_grid$total_H),
    G_conc_range          = range(total_exp_conc_grid$total_G),
    nsim                  = nsim,
    endpoint_tol          = endpoint_tol,
    endpoint_maxiter      = endpoint_maxiter,
    endpoint_damping      = endpoint_damping,
    use_optim_fallback    = use_optim_fallback,
    include_corner_points = include_corner_points,
    zero_floor_decades    = zero_floor_decades,
    trim_total_domain     = trim_total_domain,
    total_padding_decades = total_padding_decades
  )
  
  tps_result <- interpolate_HG_Tps(
    sim             = sim,
    newdata         = total_exp_conc_grid,
    padding_decades = padding_decades,
    max_train       = max_train
  )
  
  structure(
    list(
      sim        = sim,
      tps_result = tps_result
    ),
    class = "HG_titr_data"
  )
}

## Example call of simulation wrapper ------------------------------------------

if (FALSE) { ### turn TRUE to run example
  sim1 <- simulate_HG_titration(
    species_vec  = c("H1G1", "H1G2"),
    logB_vec     = c(5, 8),
    response_vec = c(0, 0, 2e5, 3e5),
    nsim         = c(100, 500),
    c_stock_H    = 1e-6,
    c_stock_G    = 1e-1,
    V_stock_H    = 3.00,
    V_G          = seq(0, 0.01, length.out = 30),
  )
  
  ggplot(data = sim1$tps_result$prediction) +
    geom_point(aes(x = total_G, y = signal_pred)) +
    theme_bw()
  
  ggplot(data = sim1$sim$surface |> filter(total_H > 9.95e-7 & total_H < 1e-6)) +
    geom_point(aes(x = -log10(free_G), y = signal)) +
    theme_bw()
}
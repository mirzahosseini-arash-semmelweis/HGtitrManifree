################################################################################
## Spectral-shift fitting for host-guest titration spectra
##
## Requires:
##   source("01_HGtitr_simulator.R")
##   source("02_HGtitr_fitter.R")
##   source("04_HGtitr_multichannel_fitter.R")
##
## Concept:
##   - equilibrium stoichiometry/logB are retained;
##   - selected species spectra are represented by smooth B-spline functions;
##   - each selected species may have its own wavelength-shift amplitude;
##   - all shifts are driven by a shared equilibrium/conformational coordinate;
##   - for fixed logB + shift parameters, spectral coefficients are profiled by
##     linear least squares, so only equilibrium/shift parameters are nonlinear.
##
## Species-specific formulation:
##
##   delta_lambda_ij = q_i * delta_lambda_max_j
##
## where q_i is either the equilibrium occupancy coordinate or a common
## free-guest Hill coordinate, and j indexes the selected shifting species.
################################################################################

HG_spectra_wide_to_long <- function(
    data,
    wavelength_cols,
    wavelength_values = NULL,
    signal_name       = "signal"
  ) {
  stopifnot(is.data.frame(data), all(c("total_H", "total_G") %in% names(data)))
  if (!all(wavelength_cols %in% names(data))) stop("Some wavelength columns are missing.")

  if (is.null(wavelength_values)) {
    wavelength_values <- suppressWarnings(as.numeric(sub("^[^0-9.+-]*", "", wavelength_cols)))
  }
  if (length(wavelength_values) != length(wavelength_cols) || any(!is.finite(wavelength_values))) {
    stop("Supply one finite numeric wavelength per wavelength column.")
  }

  out <- do.call(rbind, lapply(seq_along(wavelength_cols), function(j) {
    data.frame(
      total_H          = data$total_H,
      total_G          = data$total_G,
      wavelength       = wavelength_values[j],
      signal           = as.numeric(data[[wavelength_cols[j]]]),
      stringsAsFactors = FALSE
    )
  }))
  names(out)[names(out) == "signal"] <- signal_name
  rownames(out) <- NULL
  out
}

prepare_HG_spectral_data <- function(
    data_long,
    signal_col = "signal"
  ) {
  req <- c("total_H", "total_G", "wavelength", signal_col)
  if (!is.data.frame(data_long) || !all(req %in% names(data_long))) {
    stop("data_long must contain total_H, total_G, wavelength, and the signal column.")
  }
  dat <- data_long[, req, drop = FALSE]
  names(dat)[names(dat) == signal_col] <- "signal"
  if (any(!is.finite(as.matrix(dat))) || any(dat$total_H < 0) || any(dat$total_G < 0)) {
    stop("Spectral data must be finite and total concentrations nonnegative.")
  }

  key <- paste(sprintf("%.17g", dat$total_H), sprintf("%.17g", dat$total_G), sep = "|")
  lev <- unique(key)
  dat$point_id <- match(key, lev)
  points <- dat[!duplicated(dat$point_id), c("point_id", "total_H", "total_G")]
  points <- points[order(points$point_id), , drop = FALSE]
  dat           <- dat[order(dat$point_id, dat$wavelength), , drop = FALSE]
  rownames(dat) <- rownames(points) <- NULL
  list(data = dat, point_table = points, wavelengths = sort(unique(dat$wavelength)))
}

make_HG_spectral_basis_spec <- function(
    wavelength,
    spline_df     = 10L,
    degree        = 3L,
    max_abs_shift = 30
  ) {
  boundary <- c(min(wavelength) - abs(max_abs_shift), max(wavelength) + abs(max_abs_shift))
  B <- splines::bs(wavelength, df = spline_df, degree = degree, intercept = TRUE,
                   Boundary.knots = boundary)
  list(
    knots          = attr(B, "knots"),
    Boundary.knots = attr(B, "Boundary.knots"),
    degree         = degree,
    intercept      = TRUE,
    n_basis        = ncol(B)
  )
}

HG_eval_basis <- function(wavelength, basis_spec) {
  splines::bs(
    wavelength,
    knots          = basis_spec$knots,
    Boundary.knots = basis_spec$Boundary.knots,
    degree         = basis_spec$degree,
    intercept      = basis_spec$intercept
  )
}

HG_shift_coordinate <- function(
    equilibrium,
    total_H,
    mode         = c("occupancy_linear", "freeG_hill", "none"),
    delta_lambda = 0,
    logC50_shift = NULL,
    hill_shift   = NULL
  ) {
  mode <- match.arg(mode)
  n    <- nrow(equilibrium$concentrations)
  if (mode == "none") return(list(coordinate = rep(0, n), shift = rep(0, n)))

  if (mode == "occupancy_linear") {
    sp          <- equilibrium$species
    complex     <- !sp$species %in% c("H1G0", "H0G1")
    gcoeff      <- ifelse(complex, sp$nG, 0)
    bound_G     <- drop(equilibrium$concentrations %*% gcoeff)
    ratio_ok    <- complex & sp$nH > 0
    max_G_per_H <- max(sp$nG[ratio_ok] / sp$nH[ratio_ok])
    q <- bound_G / pmax(total_H * max_G_per_H, .Machine$double.xmin)
    q <- pmin(pmax(q, 0), 1)
    return(list(coordinate = q, shift = delta_lambda * q))
  }

  if (is.null(logC50_shift)    ||
      is.null(hill_shift)      ||
      !is.finite(logC50_shift) ||
      !is.finite(hill_shift)   ||
      hill_shift <= 0) {
    stop("freeG_hill requires finite logC50_shift and positive hill_shift.")
  }
  g <- equilibrium$concentrations[, "H0G1"]
  q <- numeric(length(g))
  pos <- g > 0
  eta <- log(10) * hill_shift * (log10(g[pos]) - logC50_shift)
  q[pos] <- plogis(eta)
  list(coordinate = q, shift = delta_lambda * q)
}

HG_build_spectral_design <- function(
    spectral_data,
    equilibrium,
    shift_matrix,
    basis_spec
) {
  dat <- spectral_data$data
  C   <- equilibrium$concentrations
  species_names <- colnames(C)

  if (!is.matrix(shift_matrix)) {
    stop("`shift_matrix` must be a numeric matrix.")
  }

  if (
    nrow(shift_matrix) != nrow(C) ||
    ncol(shift_matrix) != ncol(C)
  ) {
    stop(
      "`shift_matrix` must have one row per titration point and ",
      "one column per equilibrium species."
    )
  }

  if (is.null(colnames(shift_matrix))) {
    colnames(shift_matrix) <- species_names
  }

  if (!all(species_names %in% colnames(shift_matrix))) {
    stop("`shift_matrix` is missing one or more equilibrium species.")
  }

  shift_matrix <- shift_matrix[
    ,
    species_names,
    drop = FALSE
  ]

  blocks <- lapply(
    seq_along(species_names),
    function(j) {
      sp <- species_names[j]

      d <- shift_matrix[
        dat$point_id,
        sp
      ]

      ## Positive delta_lambda means movement to larger wavelengths.
      ## The shifted reference spectrum is therefore evaluated at
      ## lambda - delta_lambda.
      B <- HG_eval_basis(
        dat$wavelength - d,
        basis_spec
      )

      B * C[dat$point_id, j]
    }
  )

  X <- do.call(
    cbind,
    blocks
  )

  colnames(X) <- unlist(
    lapply(
      species_names,
      function(sp) {
        paste0(
          sp,
          "_b",
          seq_len(basis_spec$n_basis)
        )
      }
    ),
    use.names = FALSE
  )

  list(
    X = X,
    species_names = species_names
  )
}

HG_make_species_shift_matrix <- function(
    equilibrium,
    coordinate,
    shifted_species,
    delta_by_species
) {
  species_names <- colnames(equilibrium$concentrations)

  if (!all(shifted_species %in% species_names)) {
    stop(
      "Unknown shifted_species: ",
      paste(
        setdiff(
          shifted_species,
          species_names
        ),
        collapse = ", "
      )
    )
  }

  if (length(coordinate) != nrow(equilibrium$concentrations)) {
    stop("`coordinate` must contain one value per titration point.")
  }

  if (length(shifted_species) == 0L) {
    return(
      matrix(
        0,
        nrow = nrow(equilibrium$concentrations),
        ncol = length(species_names),
        dimnames = list(
          NULL,
          species_names
        )
      )
    )
  }

  if (
    is.null(names(delta_by_species)) ||
    !all(shifted_species %in% names(delta_by_species))
  ) {
    stop(
      "`delta_by_species` must be named for every shifted species."
    )
  }

  out <- matrix(
    0,
    nrow = nrow(equilibrium$concentrations),
    ncol = length(species_names),
    dimnames = list(
      NULL,
      species_names
    )
  )

  for (sp in shifted_species) {
    out[, sp] <- coordinate * unname(delta_by_species[sp])
  }

  out
}

HG_profile_spectra <- function(
    X,
    y,
    weights = NULL,
    ridge   = 1e-8
  ) {
  if (is.null(weights)) weights <- rep(1, length(y))
  sw <- sqrt(weights)
  Xw <- X * sw
  yw <- y * sw
  sc <- sqrt(colMeans(Xw^2))
  sc[!is.finite(sc) | sc <= 0] <- 1
  Xs <- sweep(Xw, 2, sc, "/")
  A  <- crossprod(Xs) + diag(ridge, ncol(Xs))
  b  <- crossprod(Xs, yw)
  cs <- tryCatch(solve(A, b), error = function(e) qr.solve(A, b))
  coef <- as.numeric(cs) / sc
  pred <- as.numeric(X %*% coef)
  res  <- pred - y
  list(
    coef      = coef,
    predicted = pred,
    residual  = res,
    SSE       = sum(weights * res^2),
    rank      = qr(Xs)$rank
  )
}

HG_eval_spectral_shift <- function(
    par,
    spectral_data,
    species_vec,
    basis_spec,
    shifted_species,
    shift_mode,
    weights            = NULL,
    ridge              = 1e-8,
    endpoint_tol       = 1e-7,
    endpoint_maxiter   = 500L,
    endpoint_damping   = 0.5,
    use_optim_fallback = TRUE
) {
  logB_names <- paste0(
    "logB_",
    species_vec
  )

  logB_vec <- as.numeric(par[logB_names])

  pts <- spectral_data$point_table

  eq <- solve_HG_concentrations_at_totals(
    total_H            = pts$total_H,
    total_G            = pts$total_G,
    species_vec        = species_vec,
    logB_vec           = logB_vec,
    endpoint_tol       = endpoint_tol,
    endpoint_maxiter   = endpoint_maxiter,
    endpoint_damping   = endpoint_damping,
    use_optim_fallback = use_optim_fallback
  )

  ## The equilibrium/conformational coordinate is common to all species.
  ## Species specificity enters only through delta_lambda_<species>.
  sh <- HG_shift_coordinate(
    equilibrium  = eq,
    total_H      = pts$total_H,
    mode         = shift_mode,
    delta_lambda = 1,
    logC50_shift = if (
      "logC50_shift" %in% names(par)
    ) {
      unname(
        par["logC50_shift"]
      )
    } else {
      NULL
    },
    hill_shift = if (
      "hill_shift" %in% names(par)
    ) {
      unname(
        par["hill_shift"]
      )
    } else {
      NULL
    }
  )

  delta_names <- paste0(
    "delta_lambda_",
    shifted_species
  )

  delta_by_species <- stats::setNames(
    as.numeric(par[delta_names]),
    shifted_species
  )

  shift_matrix <- HG_make_species_shift_matrix(
    equilibrium      = eq,
    coordinate       = sh$coordinate,
    shifted_species  = shifted_species,
    delta_by_species = delta_by_species
  )

  des <- HG_build_spectral_design(
    spectral_data = spectral_data,
    equilibrium   = eq,
    shift_matrix  = shift_matrix,
    basis_spec    = basis_spec
  )

  prof <- HG_profile_spectra(
    des$X,
    spectral_data$data$signal,
    weights,
    ridge
  )

  c(
    prof,
    list(
      equilibrium      = eq,
      coordinate       = sh$coordinate,
      delta_by_species = delta_by_species,
      shift_matrix     = shift_matrix,
      design           = des
    )
  )
}

fit_HG_spectral_shift <- function(
    data_long,
    species_vec,
    logB_start,
    logB_lower,
    logB_upper,
    signal_col      = "signal",
    shift_mode      = c("occupancy_linear", "freeG_hill", "none"),
    shifted_species = species_vec,

    ## Can be scalar, or a named/numeric vector with one value
    ## per shifted species.
    delta_start = 0,

    ## Common lower/upper bounds for every species-specific shift.
    delta_bounds = c(
      -60,
       60
    ),

    logC50_start  = -5,
    logC50_bounds = c(
      -9,
      -2
    ),
    hill_start  = 2,
    hill_bounds = c(
      0.25,
      8
    ),

    spline_df     = 10L,
    spline_degree = 3L,
    ridge         = 1e-8,
    weights       = NULL,

    endpoint_tol       = 1e-7,
    endpoint_maxiter   = 500L,
    endpoint_damping   = 0.5,
    use_optim_fallback = TRUE,

    nloptr_algorithm = "NLOPT_LN_BOBYQA",
    nloptr_control   = list(
      maxeval     = 1000,
      xtol_rel    = 1e-6,
      print_level = 1
    )
) {
  if (!requireNamespace("nloptr", quietly = TRUE)) {
    stop("Package nloptr is required.")
  }

  shift_mode      <- match.arg(shift_mode)
  species_vec     <- as.character(species_vec)
  shifted_species <- as.character(shifted_species)
  nsp             <- length(species_vec)

  if (
    any(
      c(
        length(logB_start),
        length(logB_lower),
        length(logB_upper)
      ) != nsp
    )
  ) {
    stop("logB start/lower/upper must match species_vec.")
  }

  if (!all(shifted_species %in% c("H1G0", "H0G1", species_vec))) {
    stop("Every shifted species must be present in the equilibrium concentration matrix.")
  }

  dat <- prepare_HG_spectral_data(
    data_long,
    signal_col
  )

  basis <- make_HG_spectral_basis_spec(
    dat$data$wavelength,
    spline_df,
    spline_degree,
    max(abs(delta_bounds))
  )

  p0 <- setNames(
    as.numeric(logB_start),
    paste0(
      "logB_",
      species_vec
    )
  )

  lo <- setNames(
    as.numeric(logB_lower),
    names(p0)
  )

  hi <- setNames(
    as.numeric(logB_upper),
    names(p0)
  )

  if (shift_mode != "none") {

    if (length(delta_start) == 1L) {
      delta_start <- rep(
        as.numeric(delta_start),
        length(shifted_species)
      )
    }

    if (length(delta_start) != length(shifted_species)) {
      stop("`delta_start` must be scalar or have one value per shifted species.")
    }

    if (
      !is.null(names(delta_start)) &&
      all(shifted_species %in% names(delta_start))
    ) {
      delta_start <- delta_start[
        shifted_species
      ]
    }

    delta_par_names <- paste0(
      "delta_lambda_",
      shifted_species
    )

    p0 <- c(
      p0,
      stats::setNames(
        as.numeric(
          delta_start
        ),
        delta_par_names
      )
    )

    lo <- c(
      lo,
      stats::setNames(
        rep(
          delta_bounds[1],
          length(shifted_species)
        ),
        delta_par_names
      )
    )

    hi <- c(
      hi,
      stats::setNames(
        rep(
          delta_bounds[2],
          length(shifted_species)
        ),
        delta_par_names
      )
    )
  }

  if (shift_mode == "freeG_hill") {
    p0 <- c(
      p0,
      logC50_shift = logC50_start,
      hill_shift   = hill_start
    )

    lo <- c(
      lo,
      logC50_shift = logC50_bounds[1],
      hill_shift   = hill_bounds[1]
    )

    hi <- c(
      hi,
      logC50_shift = logC50_bounds[2],
      hill_shift   = hill_bounds[2]
    )
  }

  ## nloptr may strip names from the optimization vector.
  parameter_names <- names(p0)

  objective <- function(
      par
  ) {
    par <- as.numeric(par)

    if (length(par) != length(parameter_names)) {
      return(
        1e100
      )
    }

    names(par) <- parameter_names

    z <- tryCatch(
      HG_eval_spectral_shift(
        par,
        dat,
        species_vec,
        basis,
        shifted_species,
        shift_mode,
        weights,
        ridge,
        endpoint_tol,
        endpoint_maxiter,
        endpoint_damping,
        use_optim_fallback
      ),
      error = function(e) {
        NULL
      }
    )

    if (
      is.null(z) ||
      !is.finite(z$SSE)
    ) {
      1e100
    } else {
      z$SSE
    }
  }

  fit <- nloptr::nloptr(
    x0     = p0,
    eval_f = objective,
    lb     = lo,
    ub     = hi,
    opts   = utils::modifyList(
      list(algorithm = nloptr_algorithm),
      nloptr_control
    )
  )

  best_par        <- fit$solution
  names(best_par) <- names(p0)

  best <- HG_eval_spectral_shift(
    best_par,
    dat,
    species_vec,
    basis,
    shifted_species,
    shift_mode,
    weights,
    ridge,
    endpoint_tol,
    endpoint_maxiter,
    endpoint_damping,
    use_optim_fallback
  )

  pred <- dat$data
  pred$predicted <- best$predicted
  pred$residual  <- best$residual

  ## Add one per-observation shift column for each shifted species.
  for (sp in shifted_species) {
    pred[[paste0("delta_lambda_", sp)]] <- best$shift_matrix[pred$point_id, sp]
  }

  wave <- dat$wavelengths

  B0 <- HG_eval_basis(
    wave,
    basis
  )

  nsb <- basis$n_basis

  species_names <- best$design$species_names

  response_spectra <- do.call(
    rbind,
    lapply(
      seq_along(species_names),
      function(j) {
        idx <- ((j - 1L) * nsb + 1L):(j * nsb)

        data.frame(
          species    = species_names[j],
          wavelength = wave,
          response   = as.numeric(B0 %*% best$coef[idx]),
          stringsAsFactors = FALSE
        )
      }
    )
  )

  shift_table            <- dat$point_table
  shift_table$coordinate <- best$coordinate

  for (sp in shifted_species) {
    shift_table[[paste0("delta_lambda_", sp)]] <- best$shift_matrix[, sp]
  }

  out <- list(
    fit      = fit,
    par      = best_par,
    logB_vec = best_par[
      paste0(
        "logB_",
        species_vec
      )
    ],
    shift_mode       = shift_mode,
    shifted_species  = shifted_species,
    delta_by_species = best$delta_by_species,
    objective_value  = best$SSE,
    linear_rank      = best$rank,
    prediction       = pred,
    response_spectra = response_spectra,
    shift_table      = shift_table,
    shift_matrix     = best$shift_matrix,
    equilibrium      = best$equilibrium,
    basis_spec       = basis,
    species_vec      = species_vec,
    spline_df        = spline_df,
    ridge            = ridge,
    data_used        = data_long
  )

  class(out) <- "HG_spectral_shift_fit"

  out
}

multistart_HG_spectral_shift <- function(
    data_long,
    species_vec,
    logB_center,
    logB_lower,
    logB_upper,
    n_start         = 30L,
    logB_jitter     = 1,
    shift_mode      = c("occupancy_linear", "freeG_hill"),
    shifted_species = species_vec,
    delta_range     = c(-60, 60),
    spline_df       = 10L,
    ridge           = 1e-8,
    seed            = 1,
    save_dir        = NULL,
    ...
) {
  shift_mode <- match.arg(shift_mode)

  set.seed(seed)

  if (!is.null(save_dir)) {
    dir.create(
      save_dir,
      recursive    = TRUE,
      showWarnings = FALSE
    )
  }

  res <- vector(
    "list",
    n_start
  )

  for (i in seq_len(n_start)) {

    lk <- logB_center +
      runif(
        length(logB_center),
        -logB_jitter,
         logB_jitter
      )

    lk <- pmin(
      pmax(
        lk,
        logB_lower
      ),
      logB_upper
    )

    ds <- stats::setNames(
      runif(
        length(shifted_species),
        delta_range[1],
        delta_range[2]
      ),
      shifted_species
    )

    cat(
      "\nStart ",
      i,
      "/",
      n_start,
      " | logB = ",
      paste(
        signif(
          lk,
          4
        ),
        collapse = ", "
      ),
      " | deltas = ",
      paste(
        paste0(
          names(ds),
          ":",
          signif(
            ds,
            4
          )
        ),
        collapse = ", "
      ),
      "\n",
      sep = ""
    )

    fit <- tryCatch(
      fit_HG_spectral_shift(
        data_long       = data_long,
        species_vec     = species_vec,
        logB_start      = lk,
        logB_lower      = logB_lower,
        logB_upper      = logB_upper,
        shift_mode      = shift_mode,
        shifted_species = shifted_species,
        delta_start     = ds,
        delta_bounds    = delta_range,
        spline_df       = spline_df,
        ridge           = ridge,
        ...
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

    file <- NA_character_

    if (!is.null(save_dir)) {
      file <- file.path(
        save_dir,
        sprintf(
          "spectral_shift_%03d.rds",
          i
        )
      )

      saveRDS(fit, file)
    }

    delta_est <- fit$delta_by_species

    row <- data.frame(
      start_id    = i,
      objective   = fit$objective_value,
      convergence = fit$fit$status,
      file        = file,
      stringsAsFactors = FALSE
    )

    for (sp in names(delta_est)) {
      row[[paste0("delta_lambda_", sp)]] <- unname(delta_est[sp])
    }

    res[[i]] <- row

    cat(
      "objective = ",
      signif(
        fit$objective_value,
        8
      ),
      "\n",
      sep = ""
    )
  }

  leaderboard <- do.call(
    rbind,
    Filter(
      Negate(is.null),
      res
    )
  )

  leaderboard <- leaderboard[
    order(leaderboard$objective),
    ,
    drop = FALSE
  ]

  rownames(leaderboard) <- NULL

  if (!is.null(save_dir)) {
    saveRDS(
      leaderboard,
      file.path(
        save_dir,
        "leaderboard.rds"
      )
    )

    write.csv(
      leaderboard,
      file.path(
        save_dir,
        "leaderboard.csv"
      ),
      row.names = FALSE
    )
  }

  leaderboard
}

################################################################################
## Example: biliverdin full CD spectra
################################################################################

if (FALSE) { ### turn TRUE to run example

  source("01_HGtitr_simulator.R")
  source("02_HGtitr_fitter.R")
  source("04_HGtitr_multichannel_fitter.R")
  source("06_HGtitr_spectral_shift_fitter_species_specific.R")

  ## Wide input example:
  ## total_H, total_G, CD340.0, CD340.1, ... CD450.0

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

  ## --------------------------------------------------------------------------
  ## H1G1 and H1G2 have independent maximum spectral shifts but share the
  ## occupancy coordinate q_i.
  ## --------------------------------------------------------------------------

  fit_bvd_shift_species <- fit_HG_spectral_shift(
    data_long = bvd_long,

    species_vec = c(
      "H1G1",
      "H1G2"
    ),

    logB_start = c(
      5.3,
      9.5
    ),

    logB_lower = c(
      2,
      4
    ),

    logB_upper = c(
      9,
      14
    ),

    shift_mode = "occupancy_linear",

    shifted_species = c(
      "H1G1",
      "H1G2"
    ),

    delta_start = c(
      H1G1 = -20,
      H1G2 = -20
    ),

    delta_bounds = c(
      -60,
       60
    ),

    spline_df = 10L,

    ridge = 1e-8,

    nloptr_control = list(
      maxeval     = 2000,
      xtol_rel    = 1e-6,
      print_level = 1
    )
  )

  fit_bvd_shift_species$par
  fit_bvd_shift_species$delta_by_species
  fit_bvd_shift_species$shift_table
  fit_bvd_shift_species$objective_value

  ## Multistart version --------------------------------------------------------

  leaderboard_bvd_shift_species <- multistart_HG_spectral_shift(
    data_long = bvd_long,

    species_vec = c(
      "H1G1",
      "H1G2"
    ),

    logB_center = c(
      5.3,
      9.5
    ),

    logB_lower = c(
      2,
      4
    ),

    logB_upper = c(
      9,
      14
    ),

    n_start = 30L,

    logB_jitter = 1.5,

    shift_mode = "occupancy_linear",

    shifted_species = c(
      "H1G1",
      "H1G2"
    ),

    delta_range = c(
      -60,
      60
    ),

    spline_df = 10L,

    ridge = 1e-8,

    seed = 1,

    save_dir = "HG_fits_biliverdin_species_specific_shift",

    nloptr_control = list(
      maxeval     = 2000,
      xtol_rel    = 1e-6,
      print_level = 0
    )
  )

  leaderboard_bvd_shift_species
}

################################################################################
## Multichannel fitting of Host-Guest titration data
##
## Shared equilibrium constants (logB_vec) + channel-specific response_vec.
##
## Prerequisites:
##   source("01_HGtitr_simulator.R")
##   source("02_HGtitr_fitter.R")
##
## The multichannel fitter deliberately solves the equilibrium directly at the
## experimental (total_H, total_G) points. It therefore does not need to rebuild
## a 2D TPS signal surface separately for each channel. For a given logB_vec,
## the equilibrium species concentrations are shared by every signal channel;
## channel predictions are then simple linear combinations of those
## concentrations through the channel-specific response_vec.
################################################################################

## Small internal helpers ------------------------------------------------------

HG_MC_safe_name <- function(x) {
  y <- gsub("[^A-Za-z0-9]+", "_", as.character(x))
  y <- gsub("^_+|_+$", "", y)
  y[y == ""] <- "channel"
  y
}

normalize_HG_signal_cols <- function(
    fit_data,
    signal_cols
) {
  if (
    !is.character(signal_cols) ||
    length(signal_cols) < 2L
  ) {
    stop(
      "`signal_cols` must be a character vector containing at least ",
      "two signal-column names."
    )
  }

  if (any(!signal_cols %in% names(fit_data))) {
    stop(
      "Signal column(s) not found in `fit_data`: ",
      paste(
        setdiff(signal_cols, names(fit_data)),
        collapse = ", "
      )
    )
  }

  channels <- names(signal_cols)

  if (
    is.null(channels) ||
    any(!nzchar(channels))
  ) {
    channels <- signal_cols
  }

  if (anyDuplicated(channels)) {
    stop("Channel names derived from `signal_cols` must be unique.")
  }

  safe_channels <- HG_MC_safe_name(channels)

  if (anyDuplicated(safe_channels)) {
    stop(
      "Channel names are not unique after sanitization. ",
      "Please supply distinct names in `signal_cols`, e.g. ",
      "c(CD = 'signal_CD', FL = 'signal_FL')."
    )
  }

  names(signal_cols) <- channels

  list(
    signal_cols   = signal_cols,
    channels      = channels,
    safe_channels = stats::setNames(
      safe_channels,
      channels
    )
  )
}

resolve_HG_channel_scale <- function(
    observed,
    channels,
    channel_scale = c("mad", "sd", "none")
) {
  if (is.character(channel_scale)) {
    channel_scale <- match.arg(channel_scale)

    scale <- switch(
      channel_scale,

      sd = apply(
        observed,
        2L,
        stats::sd
      ),

      mad = apply(
        observed,
        2L,
        stats::mad
      ),

      none = rep(
        1,
        length(channels)
      )
    )

    ## Fallback for nearly constant channels.
    bad <- !is.finite(scale) | scale <= 0

    if (any(bad)) {
      fallback <- sqrt(colMeans(observed^2))

      fallback[!is.finite(fallback) | fallback <= 0] <- 1

      scale[bad] <- fallback[bad]
    }

    names(scale) <- channels

    return(scale)
  }

  if (!is.numeric(channel_scale)) {
    stop(
      "`channel_scale` must be 'mad', 'sd', 'none', or a positive ",
      "numeric vector."
    )
  }

  if (length(channel_scale) == 1L) {
    channel_scale <- rep(
      channel_scale,
      length(channels)
    )
  }

  if (length(channel_scale) != length(channels)) {
    stop(
      "Numeric `channel_scale` must have length 1 or one value per channel."
    )
  }

  if (!is.null(names(channel_scale))) {
    if (!all(channels %in% names(channel_scale))) {
      stop(
        "Named `channel_scale` is missing channel(s): ",
        paste(
          setdiff(channels, names(channel_scale)),
          collapse = ", "
        )
      )
    }

    channel_scale <- channel_scale[channels]
  }

  channel_scale <- as.numeric(channel_scale)

  if (
    any(!is.finite(channel_scale)) ||
    any(channel_scale <= 0)
  ) {
    stop("All channel scales must be positive finite values.")
  }

  names(channel_scale) <- channels

  channel_scale
}

normalize_HG_multichannel_weights <- function(
    weights,
    n,
    channels
) {
  n_channel <- length(channels)

  if (is.null(weights)) {
    out <- matrix(
      1,
      nrow = n,
      ncol = n_channel,
      dimnames = list(
        NULL,
        channels
      )
    )

    return(out)
  }

  if (is.numeric(weights)) {

    if (length(weights) == 1L) {
      out <- matrix(
        weights,
        nrow = n,
        ncol = n_channel
      )

    } else if (length(weights) == n) {
      out <- matrix(
        rep(weights, n_channel),
        nrow = n,
        ncol = n_channel
      )

    } else if (length(weights) == n * n_channel) {
      out <- matrix(
        weights,
        nrow = n,
        ncol = n_channel
      )

    } else {
      stop(
        "Numeric `weights` must have length 1, nrow(fit_data), ",
        "or nrow(fit_data) * number_of_channels."
      )
    }

    colnames(out) <- channels

  } else if (is.matrix(weights) || is.data.frame(weights)) {

    out <- as.matrix(weights)

    if (!identical(dim(out), c(n, n_channel))) {
      stop(
        "Matrix/data-frame `weights` must have dimensions ",
        "nrow(fit_data) x number_of_channels."
      )
    }

    if (!is.null(colnames(out))) {
      if (!all(channels %in% colnames(out))) {
        stop("Weight matrix is missing one or more channel columns.")
      }

      out <- out[, channels, drop = FALSE]
    } else {
      colnames(out) <- channels
    }

  } else if (is.list(weights)) {

    if (is.null(names(weights))) {
      stop("List `weights` must be named by channel.")
    }

    missing_channels <- setdiff(
      channels,
      names(weights)
    )

    if (length(missing_channels)) {
      stop(
        "List `weights` is missing channel(s): ",
        paste(missing_channels, collapse = ", ")
      )
    }

    out <- matrix(
      NA_real_,
      nrow = n,
      ncol = n_channel,
      dimnames = list(
        NULL,
        channels
      )
    )

    for (j in seq_along(channels)) {
      ch <- channels[j]
      w  <- as.numeric(weights[[ch]])

      if (length(w) == 1L) {
        w <- rep(w, n)
      }

      if (length(w) != n) {
        stop(
          sprintf(
            "`weights[['%s']]` must have length 1 or %d.",
            ch,
            n
          )
        )
      }

      out[, j] <- w
    }

  } else {
    stop(
      "`weights` must be NULL, numeric, a matrix/data.frame, ",
      "or a named list."
    )
  }

  storage.mode(out) <- "double"

  if (
    any(!is.finite(out)) ||
    any(out < 0)
  ) {
    stop("All weights must be finite and nonnegative.")
  }

  out
}

## Multichannel fit specification ----------------------------------------------

## Format:
##
## fit_spec <- list(
##
##   logB_vec = list(
##     start     = c(5, 8),
##     lower     = c(-Inf, -Inf),
##     upper     = c( Inf,  Inf),
##     fit       = c(TRUE, TRUE),
##     map_scale = c(2, 2)
##   ),
##
##   response_vec = list(
##
##     CD = list(
##       start     = c(0, 0, 2e5, 3e5),
##       lower     = c(0, 0, -Inf, -Inf),
##       upper     = c(0, 0,  Inf,  Inf),
##       fit       = c(FALSE, FALSE, TRUE, TRUE),
##       map_scale = c(NA, NA, 2e5, 3e5)
##     ),
##
##     FL = list(
##       start     = c(0, 0, 1e5, 5e5),
##       lower     = c(0, 0, -Inf, -Inf),
##       upper     = c(0, 0,  Inf,  Inf),
##       fit       = c(FALSE, FALSE, TRUE, TRUE),
##       map_scale = c(NA, NA, 1e5, 5e5)
##     )
##   )
## )

prepare_HG_multichannel_fit_spec <- function(
    fit_spec,
    species_vec,
    channels
) {
  if (!exists("prepare_HG_fit_spec", mode = "function")) {
    stop(
      "`prepare_HG_fit_spec()` is not available. ",
      "Source 02_HGtitr_fitter.R first."
    )
  }

  if (
    !is.list(fit_spec)         ||
    is.null(fit_spec$logB_vec) ||
    is.null(fit_spec$response_vec)
  ) {
    stop(
      "`fit_spec` must contain `logB_vec` and a channel-named ",
      "`response_vec` list."
    )
  }

  if (!is.list(fit_spec$response_vec)) {
    stop("`fit_spec$response_vec` must be a list, one entry per channel.")
  }

  response_spec <- fit_spec$response_vec

  if (is.null(names(response_spec))) {
    if (length(response_spec) != length(channels)) {
      stop(
        "Unnamed `fit_spec$response_vec` must contain exactly one ",
        "entry per channel."
      )
    }

    names(response_spec) <- channels
  }

  missing_channels <- setdiff(
    channels,
    names(response_spec)
  )

  if (length(missing_channels)) {
    stop(
      "`fit_spec$response_vec` is missing channel(s): ",
      paste(missing_channels, collapse = ", ")
    )
  }

  response_spec <- response_spec[channels]

  normalized_response <- vector(
    "list",
    length(channels)
  )

  names(normalized_response) <- channels

  response_rows <- vector(
    "list",
    length(channels)
  )

  common_logB <- NULL
  logB_rows   <- NULL

  safe_channels <- stats::setNames(
    HG_MC_safe_name(channels),
    channels
  )

  if (anyDuplicated(safe_channels)) {
    stop("Channel names are not unique after sanitization.")
  }

  for (j in seq_along(channels)) {
    ch <- channels[j]

    prep <- prepare_HG_fit_spec(
      fit_spec = list(
        logB_vec     = fit_spec$logB_vec,
        response_vec = response_spec[[ch]]
      ),
      species_vec = species_vec
    )

    if (j == 1L) {
      common_logB <- prep$fit_spec$logB_vec

      logB_rows <- prep$par_table[
        prep$par_table$group == "logB_vec",
        ,
        drop = FALSE
      ]

      if (nrow(logB_rows)) {
        logB_rows$channel <- NA_character_
      }
    }

    normalized_response[[ch]] <- prep$fit_spec$response_vec

    rr <- prep$par_table[
      prep$par_table$group == "response_vec",
      ,
      drop = FALSE
    ]

    if (nrow(rr)) {
      rr$channel <- ch

      rr$name <- paste0(
        "response_",
        safe_channels[ch],
        "_",
        rr$species
      )
    }

    response_rows[[j]] <- rr
  }

  all_rows <- c(
    list(logB_rows),
    response_rows
  )

  all_rows <- Filter(
    function(x) !is.null(x) && nrow(x) > 0L,
    all_rows
  )

  if (!length(all_rows)) {
    stop("The multichannel fit specification contains no fitted parameters.")
  }

  par_tbl <- do.call(
    rbind,
    all_rows
  )

  rownames(par_tbl) <- NULL

  par_tbl <- par_tbl[
    ,
    c(
      "name",
      "group",
      "index",
      "species",
      "channel",
      "lower",
      "upper",
      "start",
      "transform",
      "map_scale",
      "map_center"
    ),
    drop = FALSE
  ]

  if (anyDuplicated(par_tbl$name)) {
    stop("Internal parameter names are not unique.")
  }

  list(
    par_table = par_tbl,

    fit_spec = list(
      logB_vec     = common_logB,
      response_vec = normalized_response
    ),

    channels      = channels,
    safe_channels = safe_channels
  )
}

par_vector_to_HG_multichannel_vectors <- function(
    par,
    par_tbl,
    fit_spec
) {
  par <- as.numeric(par)

  if (length(par) != nrow(par_tbl)) {
    stop("Parameter vector has incorrect length.")
  }

  out <- list(
    logB_vec = as.numeric(
      fit_spec$logB_vec$start
    ),

    response_vec = lapply(
      fit_spec$response_vec,
      function(x) {
        as.numeric(x$start)
      }
    )
  )

  for (i in seq_len(nrow(par_tbl))) {

    if (par_tbl$group[i] == "logB_vec") {

      out$logB_vec[par_tbl$index[i]] <- par[i]

    } else if (par_tbl$group[i] == "response_vec") {

      ch <- par_tbl$channel[i]

      out$response_vec[[ch]][par_tbl$index[i]] <- par[i]

    } else {
      stop(
        "Unknown parameter group: ",
        par_tbl$group[i]
      )
    }
  }

  out
}

## Exact equilibrium solver at experimental points -----------------------------

solve_HG_concentrations_at_totals <- function(
    total_H,
    total_G,
    species_vec,
    logB_vec,
    endpoint_tol       = 1e-7,
    endpoint_maxiter   = 500L,
    endpoint_damping   = 0.5,
    use_optim_fallback = TRUE
) {
  if (!exists("parse_species", mode = "function")) {
    stop(
      "`parse_species()` is not available. ",
      "Source 01_HGtitr_simulator.R first."
    )
  }

  if (!exists("calibrate_corner", mode = "function")) {
    stop(
      "`calibrate_corner()` is not available. ",
      "Source 01_HGtitr_simulator.R first."
    )
  }

  if (!exists("concentration_matrix", mode = "function")) {
    stop(
      "`concentration_matrix()` is not available. ",
      "Source 01_HGtitr_simulator.R first."
    )
  }

  total_H <- as.numeric(total_H)
  total_G <- as.numeric(total_G)

  if (
    length(total_H) != length(total_G) ||
    length(total_H) < 1L
  ) {
    stop("`total_H` and `total_G` must have the same positive length.")
  }

  if (
    any(!is.finite(total_H)) ||
    any(!is.finite(total_G)) ||
    any(total_H < 0)         ||
    any(total_G < 0)
  ) {
    stop("Total concentrations must be finite and nonnegative.")
  }

  if (
    !is.character(species_vec) ||
    length(species_vec) < 1L
  ) {
    stop("`species_vec` must contain at least one species.")
  }

  if (
    !is.numeric(logB_vec)                   ||
    length(logB_vec) != length(species_vec) ||
    any(!is.finite(logB_vec))
  ) {
    stop(
      "`logB_vec` must contain one finite value per complex species."
    )
  }

  parsed <- parse_species(species_vec)

  species_table <- data.frame(
    species = c(
      "H1G0",
      "H0G1",
      parsed$species
    ),

    nH = c(
      1L,
      0L,
      parsed$nH
    ),

    nG = c(
      0L,
      1L,
      parsed$nG
    ),

    logB = c(
      0,
      0,
      logB_vec
    ),

    ## Response is irrelevant for solving the equilibrium.
    response = 0,

    stringsAsFactors = FALSE
  )

  log_beta <- log(10) * species_table$logB

  pair_key <- paste(
    sprintf("%.17g", total_H),
    sprintf("%.17g", total_G),
    sep = "|"
  )

  unique_index <- !duplicated(pair_key)

  unique_H <- total_H[unique_index]

  unique_G <- total_G[unique_index]

  unique_key <- pair_key[unique_index]

  calibration_list <- lapply(
    seq_along(unique_H),
    function(i) {
      calibrate_corner(
        target_H           = unique_H[i],
        target_G           = unique_G[i],
        species            = species_table,
        log_beta           = log_beta,
        endpoint_tol       = endpoint_tol,
        endpoint_maxiter   = endpoint_maxiter,
        endpoint_damping   = endpoint_damping,
        use_optim_fallback = use_optim_fallback
      )
    }
  )

  calibration_unique <- do.call(
    rbind,
    calibration_list
  )

  rownames(calibration_unique) <- NULL

  concentration_unique <- concentration_matrix(
    H_free        = calibration_unique$free_H,
    G_free        = calibration_unique$free_G,
    species_table = species_table,
    log_beta      = log_beta
  )

  map_index <- match(
    pair_key,
    unique_key
  )

  concentrations <- concentration_unique[
    map_index,
    ,
    drop = FALSE
  ]

  calibration <- calibration_unique[
    map_index,
    ,
    drop = FALSE
  ]

  rownames(concentrations) <- NULL
  rownames(calibration)    <- NULL

  list(
    concentrations = concentrations,
    calibration    = calibration,
    species        = species_table,
    log_beta       = log_beta
  )
}

predict_HG_multichannel_at_totals <- function(
    total_H,
    total_G,
    species_vec,
    logB_vec,
    response_vec,
    endpoint_tol       = 1e-7,
    endpoint_maxiter   = 500L,
    endpoint_damping   = 0.5,
    use_optim_fallback = TRUE
) {
  if (
    !is.list(response_vec) ||
    length(response_vec) < 2L
  ) {
    stop(
      "`response_vec` must be a named list containing at least ",
      "two channel-specific response vectors."
    )
  }

  if (is.null(names(response_vec))) {
    names(response_vec) <- paste0(
      "channel",
      seq_along(response_vec)
    )
  }

  channels <- names(response_vec)

  if (
    any(!nzchar(channels)) ||
    anyDuplicated(channels)
  ) {
    stop("`response_vec` channel names must be nonempty and unique.")
  }

  expected_length <- length(species_vec) + 2L

  for (ch in channels) {
    r <- as.numeric(
      response_vec[[ch]]
    )

    if (
      length(r) != expected_length ||
      any(!is.finite(r))
    ) {
      stop(
        sprintf(
          "`response_vec[['%s']]` must contain %d finite values ",
          "in H1G0, H0G1, species_vec order.",
          ch,
          expected_length
        )
      )
    }
  }

  eq <- solve_HG_concentrations_at_totals(
    total_H            = total_H,
    total_G            = total_G,
    species_vec        = species_vec,
    logB_vec           = logB_vec,
    endpoint_tol       = endpoint_tol,
    endpoint_maxiter   = endpoint_maxiter,
    endpoint_damping   = endpoint_damping,
    use_optim_fallback = use_optim_fallback
  )

  response_matrix <- do.call(
    cbind,
    lapply(
      response_vec,
      as.numeric
    )
  )

  colnames(response_matrix) <- channels

  predicted <- eq$concentrations %*%
    response_matrix

  predicted <- as.matrix(predicted)

  colnames(predicted) <- channels

  list(
    pred          = predicted,
    equilibrium   = eq,
    logB_vec      = logB_vec,
    response_vec  = response_vec,
    channels      = channels
  )
}

## Candidate evaluation --------------------------------------------------------

simulate_HG_multichannel_fit_candidate <- function(
    par,
    par_tbl,
    fit_spec,
    fixed_params,
    total_H,
    total_G,
    verbose = TRUE
) {
  fitted_vectors <- par_vector_to_HG_multichannel_vectors(
      par      = par,
      par_tbl  = par_tbl,
      fit_spec = fit_spec
    )

  endpoint_tol <- if (
    is.null(fixed_params$endpoint_tol)
  ) {
    1e-7
  } else {
    fixed_params$endpoint_tol
  }

  endpoint_maxiter <- if (
    is.null(fixed_params$endpoint_maxiter)
  ) {
    500L
  } else {
    fixed_params$endpoint_maxiter
  }

  endpoint_damping <- if (
    is.null(fixed_params$endpoint_damping)
  ) {
    0.5
  } else {
    fixed_params$endpoint_damping
  }

  use_optim_fallback <- if (
    is.null(fixed_params$use_optim_fallback)
  ) {
    TRUE
  } else {
    fixed_params$use_optim_fallback
  }

  result <- tryCatch(

    predict_HG_multichannel_at_totals(
      total_H            = total_H,
      total_G            = total_G,
      species_vec        = fixed_params$species_vec,
      logB_vec           = fitted_vectors$logB_vec,
      response_vec       = fitted_vectors$response_vec,
      endpoint_tol       = endpoint_tol,
      endpoint_maxiter   = endpoint_maxiter,
      endpoint_damping   = endpoint_damping,
      use_optim_fallback = use_optim_fallback
    ),

    error = function(e) {
      if (verbose) {
        message(
          "Multichannel simulation failed: ",
          conditionMessage(e)
        )
      }

      NULL
    }
  )

  if (
    is.null(result) ||
    is.null(result$pred) ||
    any(!is.finite(result$pred))
  ) {
    return(NULL)
  }

  result
}

## Objective factory -----------------------------------------------------------

make_HG_multichannel_fit_objective <- function(
    fit_data,
    fixed_params,
    fit_spec,
    signal_cols,
    channel_scale = c(
      "mad",
      "sd",
      "none"
    ),
    weights      = NULL,
    loss         = c(
      "sse",
      "mae",
      "huber"
    ),
    huber_delta  = 1,
    penalty_fail = 1e100,
    verbose      = TRUE
) {
  loss <- match.arg(loss)

  if (!is.data.frame(fit_data)) {
    stop("`fit_data` must be a data.frame.")
  }

  if (
    !all(
      c("total_H", "total_G") %in%
        names(fit_data)
    )
  ) {
    stop(
      "`fit_data` must contain `total_H` and `total_G`."
    )
  }

  signal_info <- normalize_HG_signal_cols(
    fit_data    = fit_data,
    signal_cols = signal_cols
  )

  channels <- signal_info$channels

  observed <- as.matrix(
    fit_data[
      ,
      unname(signal_info$signal_cols),
      drop = FALSE
    ]
  )

  storage.mode(observed) <- "double"
  colnames(observed)     <- channels

  if (
    nrow(observed) < 2L ||
    any(!is.finite(observed))
  ) {
    stop(
      "All observed multichannel signals must be finite and ",
      "contain at least two rows."
    )
  }

  total_H <- as.numeric(fit_data$total_H)

  total_G <- as.numeric(fit_data$total_G)

  if (
    any(!is.finite(total_H)) ||
    any(!is.finite(total_G)) ||
    any(total_H < 0)         ||
    any(total_G < 0)
  ) {
    stop(
      "`fit_data$total_H` and `total_G` must be finite ",
      "and nonnegative."
    )
  }

  if (is.null(fixed_params$species_vec)) {
    stop("`fixed_params$species_vec` is required.")
  }

  prep <- prepare_HG_multichannel_fit_spec(
    fit_spec    = fit_spec,
    species_vec = fixed_params$species_vec,
    channels    = channels
  )

  par_tbl <- prep$par_table
  fit_spec_normalized <- prep$fit_spec

  scales <- resolve_HG_channel_scale(
    observed      = observed,
    channels      = channels,
    channel_scale = channel_scale
  )

  weight_matrix <- normalize_HG_multichannel_weights(
      weights  = weights,
      n        = nrow(fit_data),
      channels = channels
    )

  compute_loss <- function(res) {
    z <- as.numeric(res)

    val <- switch(
      loss,

      sse = sum(
        z^2
      ),

      mae = sum(
        abs(z)
      ),

      huber = {
        a <- abs(z)

        sum(
          ifelse(
            a <= huber_delta,
            0.5 * z^2,
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

  run_sim <- function(par) {
    simulate_HG_multichannel_fit_candidate(
      par          = par,
      par_tbl      = par_tbl,
      fit_spec     = fit_spec_normalized,
      fixed_params = fixed_params,
      total_H      = total_H,
      total_G      = total_G,
      verbose      = verbose
    )
  }

  sim_ok <- function(x) {
    !is.null(x) &&
      !is.null(x$pred) &&
      identical(
        dim(x$pred),
        dim(observed)
      ) &&
      all(is.finite(x$pred))
  }

  standardized_residual_matrix <- function(
      pred
  ) {
    sweep(
      (pred - observed) * sqrt(weight_matrix),
      2L,
      scales,
      "/"
    )
  }

  residuals <- function(par) {
    pred_obj <- run_sim(par)

    if (!sim_ok(pred_obj)) {
      return(
        rep(
          sqrt(penalty_fail / length(observed)),
          length(observed)
        )
      )
    }

    res <- standardized_residual_matrix(pred_obj$pred)

    res[!is.finite(res)] <- sqrt(penalty_fail / length(observed))

    as.numeric(res)
  }

  objective <- function(par) {
    pred_obj <- run_sim(par)

    if (!sim_ok(pred_obj)) {
      return(penalty_fail)
    }

    res <- standardized_residual_matrix(pred_obj$pred)

    if (any(!is.finite(res))) {
      return(penalty_fail)
    }

    compute_loss(res)
  }

  list(
    objective     = objective,
    residuals     = residuals,
    run_sim       = run_sim,
    fit_data      = fit_data,
    observed      = observed,
    total_H       = total_H,
    total_G       = total_G,
    signal_cols   = signal_info$signal_cols,
    channels      = channels,
    safe_channels = signal_info$safe_channels,
    channel_scale = scales,
    weights       = weight_matrix,
    par_tbl       = par_tbl,
    fit_spec      = fit_spec_normalized,
    fixed_params  = fixed_params,
    loss          = loss,
    huber_delta   = huber_delta
  )
}

## Optional checkpoint helper --------------------------------------------------

make_HG_multichannel_checkpointed_objective <- function(
    objective_unit,
    par_tbl,
    checkpoint_path,
    min_save_interval_sec = 2,
    verbose               = TRUE
) {
  best_value <- Inf
  evaluation <- 0L
  last_save  <- as.POSIXct(
    "1970-01-01",
    tz = "UTC"
  )

  checkpoint_dir <- dirname(checkpoint_path)

  if (
    !identical(checkpoint_dir, ".") &&
    !dir.exists(checkpoint_dir)
  ) {
    dir.create(
      checkpoint_dir,
      recursive    = TRUE,
      showWarnings = FALSE
    )
  }

  function(u) {
    evaluation <<- evaluation + 1L

    value <- objective_unit(u)

    if (
      is.finite(value) &&
      value < best_value
    ) {
      best_value <<- value

      now <- Sys.time()

      elapsed <- as.numeric(
        difftime(
          now,
          last_save,
          units = "secs"
        )
      )

      if (
        is.infinite(elapsed) ||
        elapsed >= min_save_interval_sec
      ) {
        par <- unit_to_par(
          u,
          par_tbl
        )

        saveRDS(
          list(
            evaluation      = evaluation,
            objective_value = value,
            u               = u,
            par             = par,
            time            = now
          ),
          checkpoint_path
        )

        last_save <<- now

        if (verbose) {
          message(
            sprintf(
              "Checkpoint: eval %d, objective %.8g",
              evaluation,
              value
            )
          )
        }
      }
    }

    value
  }
}

## Main fitting wrapper --------------------------------------------------------

fit_HG_multichannel_titration <- function(
    fit_data,
    fixed_params,
    fit_spec,
    signal_cols,
    channel_scale = c(
      "mad",
      "sd",
      "none"
    ),
    weights = NULL,
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
      maxeval     = 300,
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
  if (!exists("par_to_unit", mode = "function")) {
    stop(
      "`par_to_unit()` is not available. ",
      "Source 02_HGtitr_fitter.R first."
    )
  }

  if (!exists("unit_to_par", mode = "function")) {
    stop(
      "`unit_to_par()` is not available. ",
      "Source 02_HGtitr_fitter.R first."
    )
  }

  method <- match.arg(method)

  loss <- match.arg(loss)

  nloptr_algorithm <- match.arg(nloptr_algorithm)

  if (
    method == "minpack.lm" &&
    loss != "sse"
  ) {
    stop(
      "method = 'minpack.lm' requires loss = 'sse'."
    )
  }

  obj <- make_HG_multichannel_fit_objective(
    fit_data      = fit_data,
    fixed_params  = fixed_params,
    fit_spec      = fit_spec,
    signal_cols   = signal_cols,
    channel_scale = channel_scale,
    weights       = weights,
    loss          = loss,
    huber_delta   = huber_delta,
    penalty_fail  = penalty_fail,
    verbose       = verbose
  )

  par_tbl <- obj$par_tbl

  start <- par_tbl$start
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

  residuals_unit <- function(u) {
    obj$residuals(
      unit_to_par(
        u,
        par_tbl
      )
    )
  }

  if (is.null(checkpoint_path)) {
    objective_unit <- objective_unit_raw

  } else {
    objective_unit <-
      make_HG_multichannel_checkpointed_objective(
        objective_unit  = objective_unit_raw,
        par_tbl         = par_tbl,
        checkpoint_path = checkpoint_path,
        verbose         = verbose
      )
  }

  if (method == "nloptr") {

    if (!requireNamespace("nloptr", quietly = TRUE)) {
      stop("Package `nloptr` is required.")
    }

    fit <- nloptr::nloptr(
      x0     = u_start,
      eval_f = objective_unit,
      lb     = u_lower,
      ub     = u_upper,
      opts = utils::modifyList(
        list(
          algorithm = nloptr_algorithm
        ),
        nloptr_control
      )
    )

    best_u   <- fit$solution
    best_val <- fit$objective
  }

  if (method == "minpack.lm") {

    if (!requireNamespace("minpack.lm", quietly = TRUE)) {
      stop("Package `minpack.lm` is required.")
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
      residuals_unit(
        best_u
      )^2
    )
  }

  best_par <- unit_to_par(
    best_u,
    par_tbl
  )

  names(best_par) <- par_tbl$name

  pred_obj <- simulate_HG_multichannel_fit_candidate(
    par          = best_par,
    par_tbl      = par_tbl,
    fit_spec     = obj$fit_spec,
    fixed_params = fixed_params,
    total_H      = obj$total_H,
    total_G      = obj$total_G,
    verbose      = verbose
  )

  if (
    is.null(pred_obj) ||
    !identical(
      dim(pred_obj$pred),
      dim(obj$observed)
    )
  ) {
    stop(
      "Final multichannel simulation failed at the fitted parameter vector."
    )
  }

  fitted_vectors <- par_vector_to_HG_multichannel_vectors(
      par      = best_par,
      par_tbl  = par_tbl,
      fit_spec = obj$fit_spec
    )

  prediction_wide <- fit_data

  standardized_residual <- sweep(
    (pred_obj$pred - obj$observed) * sqrt(obj$weights),
    2L,
    obj$channel_scale,
    "/"
  )

  long_list <- vector(
    "list",
    length(obj$channels)
  )

  for (j in seq_along(obj$channels)) {
    ch      <- obj$channels[j]
    safe_ch <- obj$safe_channels[ch]

    prediction_wide[paste0("pred_", safe_ch)]     <- pred_obj$pred[, j]
    prediction_wide[paste0("residual_", safe_ch)] <- pred_obj$pred[, j] - obj$observed[, j]

    long_list[[j]] <- data.frame(
      row_id       = seq_len(nrow(fit_data)),
      total_H      = obj$total_H,
      total_G      = obj$total_G,
      channel      = ch,
      observed     = obj$observed[, j],
      predicted    = pred_obj$pred[, j],
      residual     = pred_obj$pred[, j] - obj$observed[, j],
      std_residual = standardized_residual[, j],
      stringsAsFactors = FALSE
    )
  }

  prediction_long <- do.call(
    rbind,
    long_list
  )

  rownames(prediction_long) <- NULL

  SSE_by_channel <- colSums(
    (pred_obj$pred - obj$observed)^2
  )

  RMSE_by_channel <- sqrt(
    colMeans(
      (pred_obj$pred - obj$observed)^2
    )
  )

  names(SSE_by_channel)  <- obj$channels
  names(RMSE_by_channel) <- obj$channels

  all_params <- list(
    species_vec  = fixed_params$species_vec,
    logB_vec     = fitted_vectors$logB_vec,
    response_vec = fitted_vectors$response_vec
  )

  out <- list(
    method           = method,
    fit              = fit,
    objective_value  = best_val,
    par              = best_par,
    par_table        = par_tbl,
    logB_vec         = fitted_vectors$logB_vec,
    response_vec     = fitted_vectors$response_vec,
    fit_spec         = obj$fit_spec,
    fixed_params     = fixed_params,
    all_params       = all_params,
    signal_cols      = obj$signal_cols,
    channels         = obj$channels,
    channel_scale    = obj$channel_scale,
    weights          = obj$weights,
    prediction       = prediction_wide,
    prediction_wide  = prediction_wide,
    prediction_long  = prediction_long,
    observed_matrix  = obj$observed,
    predicted_matrix = pred_obj$pred,
    equilibrium      = pred_obj$equilibrium,
    SSE_by_channel   = SSE_by_channel,
    RMSE_by_channel  = RMSE_by_channel,
    loss             = loss,
    data_used        = fit_data
  )

  class(out) <- "HG_multichannel_titration_fit"

  out
}

## Convenience summaries -------------------------------------------------------

summary_HG_multichannel_fit <- function(
    fit
) {
  if (!inherits(fit, "HG_multichannel_titration_fit")) {
    stop("`fit` must be an HG_multichannel_titration_fit object.")
  }

  list(
    objective_value = fit$objective_value,
    logB_vec        = fit$logB_vec,
    response_vec    = fit$response_vec,
    SSE_by_channel  = fit$SSE_by_channel,
    RMSE_by_channel = fit$RMSE_by_channel,
    channel_scale   = fit$channel_scale
  )
}

## Example ---------------------------------------------------------------------

if (FALSE) { ### turn TRUE to run example
  ## --------------------------------------------------------------------------
  ## 0. Source prerequisites
  ## --------------------------------------------------------------------------
  source("01_HGtitr_simulator.R")
  source("02_HGtitr_fitter.R")
  source("04_HGtitr_multichannel_fitter.R")
  library(ggplot2)

  ## --------------------------------------------------------------------------
  ## 1. Generate a common titration design
  ## --------------------------------------------------------------------------
  conc_grid <- make_HG_titration_grid(
    c_stock_H = 1e-6,
    c_stock_G = 1e-1,
    V_stock_H = 3.00,
    V_G       = seq(
      0,
      0.01,
      length.out = 30
    )
  )

  species_true <- c(
    "H1G1",
    "H1G2"
  )

  logB_true <- c(
    5,
    5.5
  )

  response_true <- list(
    CD = c(
      0,
      0,
      2.0e5,
      3.0e5
    ),

    FL = c(
      0,
      0,
      -1.0e5,
      5.0e5
    )
  )

  truth <- predict_HG_multichannel_at_totals(
    total_H      = conc_grid$total_H,
    total_G      = conc_grid$total_G,
    species_vec  = species_true,
    logB_vec     = logB_true,
    response_vec = response_true
  )

  ## Channel-specific measurement noise.
  noise_sd <- c(
    CD = 1e-3,
    FL = 2e-3
  )

  set.seed(1)

  fit_data_mc <- data.frame(
    total_H = conc_grid$total_H,
    total_G = conc_grid$total_G,

    signal_CD =
      truth$pred[, "CD"] +
      rnorm(
        nrow(conc_grid),
        sd = noise_sd["CD"]
      ),

    signal_FL =
      truth$pred[, "FL"] +
      rnorm(
        nrow(conc_grid),
        sd = noise_sd["FL"]
      )
  )

  ## --------------------------------------------------------------------------
  ## 2. Fixed chemistry
  ##
  ## Only species_vec is essential for the direct multichannel fitter.
  ## The endpoint settings below are optional and default to the same values
  ## as the single-channel simulator.
  ## --------------------------------------------------------------------------
  fixed_params_mc <- list(
    species_vec        = species_true,
    endpoint_tol       = 1e-7,
    endpoint_maxiter   = 500L,
    endpoint_damping   = 0.5,
    use_optim_fallback = TRUE
  )

  ## --------------------------------------------------------------------------
  ## 3. Shared-logB / channel-specific-response specification
  ## --------------------------------------------------------------------------
  fit_spec_mc <- list(

    logB_vec = list(
      start     = c(4.5, 7.5),
      lower     = c(-Inf, -Inf),
      upper     = c( Inf,  Inf),
      fit       = c(TRUE, TRUE),
      map_scale = c(2, 2)
    ),

    response_vec = list(

      CD = list(
        start     = c(0, 0, 1.5e5, 2.5e5),
        lower     = c(0, 0, -Inf, -Inf),
        upper     = c(0, 0,  Inf,  Inf),
        fit       = c(FALSE, FALSE, TRUE, TRUE),
        map_scale = c(NA, NA, 2e5, 3e5)
      ),

      FL = list(
        start     = c(0, 0, -0.5e5, 4.0e5),
        lower     = c(0, 0, -Inf, -Inf),
        upper     = c(0, 0,  Inf,  Inf),
        fit       = c(FALSE, FALSE, TRUE, TRUE),
        map_scale = c(NA, NA, 1e5, 5e5)
      )
    )
  )

  ## --------------------------------------------------------------------------
  ## 4. Deterministic joint fit
  ##
  ## For channels with different units/noise, numeric channel_scale is
  ## preferable. If noise is unknown, channel_scale = "sd" is a useful
  ## optimization default that prevents a numerically large channel from
  ## dominating the fit.
  ## --------------------------------------------------------------------------
  fit_mc <- fit_HG_multichannel_titration(
    fit_data     = fit_data_mc,
    fixed_params = fixed_params_mc,
    fit_spec     = fit_spec_mc,

    signal_cols = c(
      CD = "signal_CD",
      FL = "signal_FL"
    ),

    channel_scale = noise_sd,

    method           = "nloptr",
    nloptr_algorithm = "NLOPT_LN_BOBYQA",
    loss             = "sse",

    nloptr_control = list(
      maxeval     = 250,
      xtol_rel    = 1e-5,
      print_level = 1
    ),

    verbose = TRUE
  )

  fit_mc$logB_vec
  fit_mc$response_vec
  fit_mc$SSE_by_channel
  fit_mc$RMSE_by_channel

  ## --------------------------------------------------------------------------
  ## 5. Plot all channels
  ## --------------------------------------------------------------------------

  ggplot(
    fit_mc$prediction_long,
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
    fit_mc$prediction_long,
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
  
  ## --------------------------------------------------------------------------
  ## +1. Try with H1G1a and H1G1b
  ## --------------------------------------------------------------------------
  conc_grid <- make_HG_titration_grid(
    c_stock_H = 1e-6,
    c_stock_G = 1e-1,
    V_stock_H = 3.00,
    V_G       = seq(
      0,
      0.01,
      length.out = 30
    )
  )
  
  species_true <- c(
    "H1G1a",
    "H1G1b"
  )
  
  logB_true <- c(
    5,
    8
  )
  
  response_true <- list(
    CD = c(
      0,
      0,
      2.0e5,
      3.0e5
    ),
    
    FL = c(
      0,
      0,
      -1.0e5,
      5.0e5
    )
  )
  
  truth <- predict_HG_multichannel_at_totals(
    total_H      = conc_grid$total_H,
    total_G      = conc_grid$total_G,
    species_vec  = species_true,
    logB_vec     = logB_true,
    response_vec = response_true
  )
  
  ## Channel-specific measurement noise.
  noise_sd <- c(
    CD = 1e-3,
    FL = 2e-3
  )
  
  set.seed(1)
  
  fit_data_mc <- data.frame(
    total_H = conc_grid$total_H,
    total_G = conc_grid$total_G,
    
    signal_CD =
      truth$pred[, "CD"] +
      rnorm(
        nrow(conc_grid),
        sd = noise_sd["CD"]
      ),
    
    signal_FL =
      truth$pred[, "FL"] +
      rnorm(
        nrow(conc_grid),
        sd = noise_sd["FL"]
      )
  )
  
  fixed_params_mc <- list(
    species_vec        = species_true,
    endpoint_tol       = 1e-7,
    endpoint_maxiter   = 500L,
    endpoint_damping   = 0.5,
    use_optim_fallback = TRUE
  )
  
  fit_spec_mc <- list(
    
    logB_vec = list(
      start     = c(4.5, 7.5),
      lower     = c(-Inf, -Inf),
      upper     = c( Inf,  Inf),
      fit       = c(TRUE, TRUE),
      map_scale = c(2, 2)
    ),
    
    response_vec = list(
      
      CD = list(
        start     = c(0, 0, 1.5e5, 2.5e5),
        lower     = c(0, 0, -Inf, -Inf),
        upper     = c(0, 0,  Inf,  Inf),
        fit       = c(FALSE, FALSE, TRUE, TRUE),
        map_scale = c(NA, NA, 2e5, 3e5)
      ),
      
      FL = list(
        start     = c(0, 0, -0.5e5, 4.0e5),
        lower     = c(0, 0, -Inf, -Inf),
        upper     = c(0, 0,  Inf,  Inf),
        fit       = c(FALSE, FALSE, TRUE, TRUE),
        map_scale = c(NA, NA, 1e5, 5e5)
      )
    )
  )
  
  fit_mc <- fit_HG_multichannel_titration(
    fit_data     = fit_data_mc,
    fixed_params = fixed_params_mc,
    fit_spec     = fit_spec_mc,
    
    signal_cols = c(
      CD = "signal_CD",
      FL = "signal_FL"
    ),
    
    channel_scale = noise_sd,
    
    method           = "nloptr",
    nloptr_algorithm = "NLOPT_LN_BOBYQA",
    loss             = "sse",
    
    nloptr_control = list(
      maxeval     = 250,
      xtol_rel    = 1e-5,
      print_level = 1
    ),
    
    verbose = TRUE
  )
  
  fit_mc$logB_vec
  fit_mc$response_vec
  fit_mc$SSE_by_channel
  fit_mc$RMSE_by_channel
  
  ggplot(
    fit_mc$prediction_long,
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
    fit_mc$prediction_long,
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
}

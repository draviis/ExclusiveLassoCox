GLM_FAMILIES <- c(gaussian=0,
                  binomial=1,
                  poisson=2,
				  cox=3)

#' Fit a GLM with Exclusive Lasso Regularization
#'
#' Fit a generalized linear model via maximum penalized likelihood
#' using the exclusive lasso penalty. The regularization path is computed
#' along a grid of values for the regularization parameter (lmd).
#' The interface is intentionally similar to that of \code{\link[glmnet]{glmnet}} in
#' the package of the same name.
#'
#' Note that unlike Campbell and Allen (2017), we use the "1/n"-scaling of the
#' loss function.
#'
#' For the Gaussian case:
#' \deqn{\frac{1}{2n}|y - X\beta|_2^2 + \lmd P(\beta, G)}
#'
#' For other GLMs:
#' \deqn{-\frac{1}{n}\ell(y, X\beta)+ \lmd P(\beta, G)}
#'
#' @param X The matrix of predictors (\eqn{X \in \R^{n \times p}}{X})
#' @param y The response vector (\eqn{y})
#' @param groups An integer vector of length \eqn{p} indicating group membership.
#'     (Cf. the \code{group} argument of \code{\link[grpreg]{grpreg}})
#' @param family The GLM response type. (Cf. the \code{family} argument of
#'               \code{\link[stats]{glm}})
#' @param weights Weights applied to individual
#'     observations. If not supplied, all observations will be equally
#'     weighted. Will be re-scaled to sum to \eqn{n} if
#'     necessary. (Cf. the \code{weight} argument of
#'     \code{\link[stats]{lm}})
#' @param offset A vector of length \eqn{n} included in the linear
#'     predictor.
#' @param nlmd The number of lmd values to use in computing the
#'    regularization path. Note that the time to run is typically sublinear
#'    in the grid size due to the use of warm starts.
#' @param lmd.min.ratio The smallest value of lmd to be used, as a fraction
#'      of the largest value of lmd used. Unlike the lasso, there is no
#'      value of lmd such that the solution is wholly sparse, but we still
#'      use lmd_max from the lasso.
#' @param lmd A user-specified sequence of lmds to use.
#' @param standardize Should \code{X} be centered and scaled before fitting?
#' @param intercept Should the fitted model have an (unpenalized) intercept term?
#' @param thresh The convergence threshold used for the proximal
#'    gradient or coordinate-descent algorithm used to solve the
#'    penalized regression problem.
#' @param thresh_prox The convergence threshold used for the
#'    coordinate-descent algorithm used to evaluate the proximal operator.
#' @param lower.limits A vector of lower bounds for each coefficient (default \code{-Inf}).
#'                     Can either be a scalar (applied to each coefficient) or a vector
#'                     of length \code{p} (number of coefficients).
#' @param upper.limits A vector of lower bounds for each coefficient (default \code{Inf}).
#'                     Can either be a scalar (applied to each coefficient) or a vector
#'                     of length \code{p} (number of coefficients).
#' @param skip_df Should the DF calculations be skipped? They are often slower
#'    than the actual model fitting; if calling \code{exclusive_lasso} repeatedly
#'    it may be useful to skip these calculations.
#' @param algorithm Which algorithm to use, proximal gradient (\code{"pg"}) or
#'    coordinate descent (\code{"cd"})? Empirically, coordinate descent appears
#'    to be faster for most problems (consistent with Campbell and Allen), but
#'    proximal gradient may be faster for certain problems with many small groups
#'    where the proximal operator may be evaluated quickly and to high precision.
#' @details By default, an optimized implementation is used for \code{family="gaussian"}
#'          which is approximately 2x faster for most problems. If you wish
#'          to disable this code path and use the standard GLM implementation
#'          with Gaussian response, set \code{options(ExclusiveLasso.gaussian_fast_path=FALSE).}
#' @include internals.R
#' @examples
#' n <- 200
#' p <- 500
#' groups <- rep(1:10, times=50)
#' beta <- numeric(p);
#' beta[1:10] <- 3
#'
#' X <- matrix(rnorm(n * p), ncol=p)
#' y <- X %*% beta + rnorm(n)
#'
#' exfit <- exclusive_lasso(X, y, groups)
#' @importFrom stats median weighted.mean
#' @importMethodsFrom Matrix colMeans colSums
#' @importClassesFrom Matrix dgCMatrix
#' @return An object of class \code{ExclusiveLassoFit} containing \itemize{
#' \item \code{coef} - A matrix of estimated coefficients
#' \item \code{intercept} - A vector of estimated intercepts if \code{intercept=TRUE}
#' \item \code{X, y, groups, weights, offset} - The data used to fit the model
#' \item \code{lmd} - The vector of \eqn{\lmd}{lmd} used
#' \item \code{df} - An unbiased estimate of the degrees of freedom (see Theorem
#'       5 in [1])
#' \item \code{nnz} - The number of non-zero coefficients at each value of
#'       \eqn{\lmd}{lmd}
#' }
#' @references
#' Campbell, Frederick and Genevera I. Allen. "Within Group Variable Selection
#'     with the Exclusive Lasso". Electronic Journal of Statistics 11(2),
#'     pp.4220-4257. 2017. \doi{10.1214/17-EJS1317}
#' @export
exclusive_lasso <- function(X, y, groups, family=c("gaussian", "binomial", "poisson","cox"),
                            weights, offset, nlmd=100,
                            lmd.min.ratio=ifelse(nobs < nvars, 0.01, 1e-04),
                            lmd, standardize=TRUE, intercept=TRUE,
                            lower.limits = rep(-Inf, nvars),
                            upper.limits = rep(Inf, nvars),
                            thresh=1e-07, thresh_prox=thresh,
                            skip_df=FALSE,
                            algorithm=c("cd", "pg")){

    tic <- Sys.time()

    ####################
    ##
    ## Input Validation
    ##
    ####################

    nobs <- NROW(X);
    nvars <- NCOL(X);

    #nobs <- NROW(X)  # Number of rows in X
    
    # Check if y matches the number of rows in X
    if (is.numeric(y)) {
      # Gaussian case: y must have the same length as rows in X
      if (length(y) != nobs) {
        stop(sQuote("NROW(X)"), " and ", sQuote("length(y)"), " must match for numeric y.")
      }
    } else if (inherits(y, "Surv")) {
      # Survival case: y must have the same number of rows as X
      if (NROW(y) != nobs) {
        stop(sQuote("NROW(X)"), " and ", sQuote("NROW(y)"), " must match for Surv y.")
      }
    } else {
      # Handle unsupported y types
      stop("Unsupported type for response variable 'y'. Must be numeric or Surv object.")
    }
    

    if(length(groups) != nvars){
        stop(sQuote("NCOL(X)"), " and ", sQuote("length(groups)"), " must match.")
    }

    if(anyNA(X) || anyNA(y)){
        stop(sQuote("exclusive_lasso"), " does not support missing data.")
    }

    if(!all(is.finite(X))){
        stop("All elements of ", sQuote("X"), " must be finite.")
    }

    if(!all(is.finite(y))){
        stop("All elements of ", sQuote("y"), " must be finite.")
    }

    ## Index groups from 0 to `num_unique(groups) - 1` to represent
    ## in a arma::ivec
    groups <- match(groups, unique(groups)) - 1

    family <- match.arg(family)

	if (family == "cox") {
		# if (!all(c("time", "event") %in% colnames(y))) {
		# 	stop("For Cox regression, `y` must be a dataset with columns `time` and `event`.")
		# }
		if (any(y[,1] < 0)) {
			stop("`time` in `y` must contain non-negative values for Cox regression.")
		}
		if (!all(y[,2] %in% c(0, 1))) {
			stop("`event` in `y` must be binary (0 or 1) for Cox regression.")
		}
	}
			
	
	if(family == "poisson"){
        if(any(y < 0)){
            stop(sQuote("y"), " must be non-negative for Poisson regression.")
        }
    }


    if(family == "binomial"){
        if(any(y < 0) || any(y > 1)){
            stop(sQuote("y"), " must be in [0, 1] for logistic regression.")
        }
    }

    if(missing(weights)){
        weights <- rep(1, nobs)
    }

    if(length(weights) != nobs){
        stop(sQuote("NROW(X)"), " and ", sQuote("length(weights)"), " must match.")
    }

    if(any(weights <= 0)){
        stop("Observation weights must be strictly positive.")
    }

    if(sum(weights) != nobs){
        weights <- weights * nobs / sum(weights)
        warning(sQuote("sum(weights)"), " is not equal to ", sQuote("NROW(X)."), " Renormalizing...")
    }

    if(missing(offset)){
        offset <- rep(0, nobs)
    }

    if(length(offset) != nobs){
        stop(sQuote("NROW(X)"), " and ", sQuote("length(offset)"), " must match.")
    }

    nlmd <- as.integer(nlmd)

    if((lmd.min.ratio <= 0) || (lmd.min.ratio >= 1)){
        stop(sQuote("lmd.min.ratio"), " must be in the interval (0, 1).")
    }

    if(standardize){
        ## FIXME -- This form of standardizing X isn't quite right with observation weights
        Xsc <- scale(X, center=TRUE, scale=TRUE)
        X_scale <- attr(Xsc, "scaled:scale", exact=TRUE)
        X_center <- attr(Xsc, "scaled:center", exact=TRUE)

        if(!all(is.finite(Xsc))){
            stop("Non-finite ", sQuote("X"), " found after standardization.")
        }
    } else {
        Xsc <- X
        X_scale <- rep(1, nvars)
        X_center <- rep(0, nvars)
    }

	if (missing(lmd)) {
		# If the family is "cox" (Surv object)
		if (family == "cox") {
			lmd_max <-1
			# Calculate baseline hazard at beta = 0
			#hazards <- 1 / rowSums(exp(Xsc))  # Assuming Xsc is standardized

			# Deviance residuals: delta_i - baseline hazard
			#residuals <- y[,2] - hazards  # status is the event indicator (1 for event, 0 otherwise)

			# Calculate lambda max
			#lmd_max <- max(abs(crossprod(Xsc, residuals))) / nobs  # nobs is the number of observations

			# Create a sequence of lambda values based on lmd_max
			lmd <- logspace(lmd.min.ratio * lmd_max, lmd_max, length.out = nlmd)
			
		} else {
			# For other families (binomial, gaussian, poisson, etc.)
			lmd_max <- max(abs(crossprod(Xsc, y - offset - weighted.mean(y, weights) * intercept) / nobs))
			
			# Check the value of lmd_max (it should be scalar)
			if (length(lmd_max) != 1) {
				stop("lmd_max should be a scalar, but got: ", length(lmd_max))
			}
			
			lmd <- logspace(lmd.min.ratio * lmd_max, lmd_max, length.out = nlmd)
		}
	}




    if(length(lmd) < 1){
        stop("Must solve for at least one value of lmd.")
    }

    if(any(lmd <= 0)){
        stop("All values of ", sQuote("lmd"), " must be positive.")
    }

    if(is.unsorted(lmd)){
        warning("User-supplied ", sQuote("lmd"), " is not increasing. Sorting for maximum performance.")
        lmd <- sort(lmd)
    }

    if(thresh_prox <= 0){
        stop(sQuote("thresh_prox"), " must be positive.")
    }

    if(thresh <= 0){
        stop(sQuote("thresh"), " must be positive.")
    }

    if(any(is.na(upper.limits)) || any(is.nan(upper.limits))){
        stop(sQuote("upper.limits"), " should be either finite or +/-Inf.")
    }

    if(any(is.na(upper.limits)) || any(is.nan(lower.limits))){
        stop(sQuote("lower.limits"), " should be either finite or +/-Inf.")
    }

    if(length(lower.limits) == 1L){
        lower.limits <- rep(lower.limits, nvars)
    }

    if(length(upper.limits) == 1L){
        upper.limits <- rep(upper.limits, nvars)
    }

    if(length(upper.limits) != nvars){
        stop(sQuote("upper.limits"), " must be of length ", sQuote("NCOL(X)."))
    }

    if(length(lower.limits) != nvars){
        stop(sQuote("lower.limits"), " must be of length ", sQuote("NCOL(X)."))
    }

    if(any(upper.limits <= lower.limits)){
        stop(sQuote("upper.limits"), " must be strictly greater than ", sQuote("lower.limits."))
    }

    algorithm <- match.arg(algorithm)


	
	if ((family == "gaussian") && getOption("ExclusiveLasso.gaussian_fast_path", TRUE)) {
		if (algorithm == "cd") {
			res <- exclusive_lasso_gaussian_cd(X = Xsc, y = y, groups = groups,
                                           lmd = lmd, w = weights, o = offset,
                                           lower_bound = lower.limits, upper_bound = upper.limits,
                                           thresh = thresh, intercept = intercept)
		} else {
			res <- exclusive_lasso_gaussian_pg(X = Xsc, y = y, groups = groups,
                                           lmd = lmd, w = weights, o = offset,
                                           lower_bound = lower.limits, upper_bound = upper.limits,
                                           thresh_prox = thresh_prox, thresh = thresh,
                                           intercept = intercept)
		}
	} else if (family == "cox") {
		# Extract time and event from Surv object
		time <- y[, 1]
		event <- y[, 2]

		if (algorithm == "cd") {
			res <- exclusive_lasso_cox_cd(X = Xsc, time = time, event = event, groups = groups,
                                      lmd = lmd, w = weights, o = offset,
                                      lower_bound = lower.limits, upper_bound = upper.limits,
                                      thresh = thresh, intercept = intercept)
		} else {
			# You can add the handling for `pg` algorithm if needed for Cox
			# res <- exclusive_lasso_cox_pg(...)
		}
	} else {
		if (algorithm == "cd") {
			res <- exclusive_lasso_glm_cd(X = Xsc, y = y, groups = groups,
                                      lmd = lmd, w = weights, o = offset,
                                      family = GLM_FAMILIES[family],
                                      lower_bound = lower.limits, upper_bound = upper.limits,
                                      thresh = thresh, thresh_prox = thresh_prox,
                                      intercept = intercept)
		} else {
			res <- exclusive_lasso_glm_pg(X = Xsc, y = y, groups = groups,
                                      lmd = lmd, w = weights, o = offset,
                                      family = GLM_FAMILIES[family],
                                      lower_bound = lower.limits, upper_bound = upper.limits,
                                      thresh = thresh, thresh_prox = thresh_prox,
                                      intercept = intercept)
		}
	}


    ## Convert intercept to R vector (arma::vec => R column vector)
    res$intercept <- as.vector(res$intercept)

    ## Convert coefficients and intercept back to original scale
    if(standardize){
        ## To get back to the original X, we multiply by X_scale,
        ## so we divide beta to keep things on the same unit
        res$coef <- res$coef / X_scale
        if(intercept){
            ## Map back to original X (undo scale + center)
            ##
            ## We handled the scaling above, now we adjust for the
            ## centering of X: beta(X - colMeans(X)) = beta * X - beta * colMeans(X)
            ## To uncenter we add back in beta * colMeans(X), summed over all observations
            res$intercept <- res$intercept - colSums(res$coef * X_center)
        }
    }

    ## Degrees of freedom -- calculated using original scale matrix
    ## (though it shouldn't really matter)
    if(!skip_df){
        df <- calculate_exclusive_lasso_df(X, lmd, groups, res$coef)
    } else {
        df <- NULL
    }

    if(!is.null(colnames(X))){
        rownames(res$coef) <- colnames(X)
    }

    result <- list(coef=res$coef,
                   intercept=res$intercept,
                   y=y,
                   X=X,
                   standardize=standardize,
                   groups=groups,
                   lmd=lmd,
                   weights=weights,
                   offset=offset,
                   family=family,
                   df=df,
                   algorithm=algorithm,
                   nnz=apply(res$coef, 2, function(x) sum(x != 0)),
                   time=Sys.time() - tic)

    class(result) <- c("ExclusiveLassoFit", class(result))

    result
}

has_intercept <- function(x){
    !is.null(x$intercept)
}

has_offset <- function(x){
    any(x$offset != 0)
}

#' @export
print.ExclusiveLassoFit <- function(x, ..., indent=0){
    icat("Exclusive Lasso Fit", "\n", indent=indent)
    icat("-------------------", "\n", indent=indent)
    icat("\n", indent=indent)
    icat("N: ", NROW(x$X), ". P: ", NCOL(x$X), ".\n", sep="", indent=indent)
    icat(length(unique(x$groups)), "groups. Median size", median(table(x$groups)), "\n", indent=indent)
    icat("\n", indent=indent)
    icat("Grid:", length(x$lmd), "values of lmd. \n", indent=indent)
    icat("  Miniumum:", min(x$lmd), "\n", indent=indent)
    icat("  Maximum: ", max(x$lmd), "\n", indent=indent)
    if(!is.null(x$df)){
      icat("  Degrees of freedom: ", min(x$df), " --> ", max(x$df), "\n", indent=indent)
    }
    icat("  Number of selected variables:", min(x$nnz), " --> ", max(x$nnz), "\n", indent=indent)
    icat("\n", indent=indent)
    icat("Fit Options:\n", indent=indent)
    icat("  - Family:        ", capitalize_string(x$family), "\n", indent=indent)
    icat("  - Intercept:     ", has_intercept(x), "\n", indent=indent)
    icat("  - Standardize X: ", x$standardize, "\n", indent=indent)
    icat("  - Algorithm:     ", switch(x$algorithm, pg="Proximal Gradient", cd="Coordinate Descent"),
         "\n", indent=indent)
    icat("\n", indent=indent)
    icat("Time: ", sprintf("%2.3f %s", x$time, attr(x$time, "units")), "\n", indent=indent)
    icat("\n", indent=indent)

    invisible(x)
}

# Refit exclussive lasso on new lmd grid
# Used internally by predict(exact=TRUE)
#' @importFrom utils modifyList
update_fit <- function(object, lmd, ...){
    ARGS <- list(X=object$X,
                 y=object$y,
                 groups=object$groups,
                 weights=object$weights,
                 offset=object$offset,
                 family=object$gamily,
                 standardize=object$standardize,
                 intercept=has_intercept(object),
                 lmd=lmd)

    ARGS <- modifyList(ARGS, list(...))

    do.call(exclusive_lasso, ARGS)
}

#' @rdname predict.ExclusiveLassoFit
#' @export
#' @importFrom stats predict
coef.ExclusiveLassoFit <- function(object, lmd=s, s=NULL,
                                   exact=FALSE, group_threshold=FALSE, ...){

    predict(object, lmd=lmd, type="coefficients",
            exact=exact, group_threshold=group_threshold, ...)
}

#' Make predictions using the exclusive lasso.
#'
#' Make predictions using the exclusive lasso. Similar to \code{\link[glmnet]{predict.glmnet}}.
#' \code{coef(...)} is a wrapper around \code{predict(..., type="coefficients")}.
#'
#' @rdname predict.ExclusiveLassoFit
#' @importFrom Matrix Matrix
#' @export
#' @param object An \code{ExclusiveLassoFit} object produced by \code{\link{exclusive_lasso}}.
#' @param newx New data \eqn{X \in R^{m \times p}}{X} on which to make predictions. If not
#'    supplied, predictions are made on trainng data.
#' @param s An alternate argument that may be used to supply \code{lmd}. Included for
#'    compatability with \code{\link[glmnet]{glmnet}}.
#' @param lmd The value of the regularization paramter (\eqn{lmd}) at which to
#'    return the fitted coefficients or predicted values. If not supplied, results for
#'    the entire regularization path are returned. Can be a vector.
#' @param type The type of "prediction" to return. If \code{type="link"}, returns
#'    the linear predictor. If \code{type="response"}, returns the expected
#'    value of the response. If \code{type="coefficients"}, returns the coefficients
#'    used to calculate the linear predictor. (Cf. the \code{type} argument
#'    of \code{\link[glmnet]{predict.glmnet}})
#' @param exact Should the exclusive lasso be re-run for provided values of \code{lmd}?
#'    If \code{FALSE}, approximate values obtained by linear interpolation on grid points
#'    are used instead. (Cf. the \code{exact} argument of \code{\link[glmnet]{predict.glmnet}})
#' @param offset An offset term used in predictions. If not supplied, all offets are
#'    taken to be zero. If the original fit was made with an offset, \code{offset} will
#'    be required.
#' @param group_threshold If \code{TRUE}, (hard-)threshold coefficients so that
#'    there is exactly one non-zero coefficient in each group.
#' @param ... Additional arguments passed to \code{\link{exclusive_lasso}} if
#'    \code{exact=TRUE} and ignored otherwise.
#' @examples
#' n <- 200
#' p <- 500
#' groups <- rep(1:10, times=50)
#' beta <- numeric(p);
#' beta[1:10] <- 3
#'
#' X <- matrix(rnorm(n * p), ncol=p)
#' y <- X %*% beta + rnorm(n)
#'
#' exfit <- exclusive_lasso(X, y, groups)
#' coef(exfit, lmd=1)
#' predict(exfit, lmd=1, newx = -X)
predict.ExclusiveLassoFit <- function(object, newx, lmd=s, s=NULL,
                                      type=c("link", "response", "coefficients"),
                                      group_threshold=FALSE,
                                      exact=FALSE, offset, ...){
    type <- match.arg(type)

    ## Get coefficients first
    if(!is.null(lmd)){
        if(exact){
            object <- update_fit(object, lmd=lmd, ...)

            if(has_intercept(object)){
                int <- Matrix(object$intercept, nrow=1,
                              sparse=TRUE, dimnames=list("(Intercept)", NULL))
            } else {
                int <- Matrix(0, nrow=1, ncol=length(object$lmd),
                              sparse=TRUE, dimnames=list("(Intercept)", NULL))
            }

            coef <- rbind(int, object$coef)
        } else {
            if(has_intercept(object)){
                int <- Matrix(object$intercept, nrow=1,
                              sparse=TRUE, dimnames=list("(Intercept)", NULL))
            } else {
                int <- Matrix(0, nrow=1, ncol=length(object$lmd),
                              sparse=TRUE, dimnames=list("(Intercept)", NULL))
            }

            coef <- rbind(int, object$coef)
            lmd <- clamp(lmd, range=range(object$lmd))

            coef <- lmd_interp(coef,
                                  old_lmd=object$lmd,
                                  new_lmd=lmd)
        }
    } else {
        if(has_intercept(object)){
             int <- Matrix(object$intercept, nrow=1,
                           sparse=TRUE, dimnames=list("(Intercept)", NULL))
        } else {
             int <- Matrix(0, nrow=1, ncol=length(object$lmd),
                           sparse=TRUE, dimnames=list("(Intercept)", NULL))
        }

        coef <- rbind(int, object$coef)
    }

    if(group_threshold){
            coef[-1,,drop=FALSE] <- Matrix(apply(coef[-1,,drop=FALSE], 2, do_group_threshold, object$groups), sparse=TRUE)
    }

    if(type == "coefficients"){
        return(coef) ## Done
    }

    if(missing(newx)){
        link <- object$offset + cbind(1, object$X) %*% coef
    } else {
        if(missing(offset)){
            if(has_offset(object)){
                stop("Original fit had an offset term but", sQuote("offset"), "not supplied.")
            } else {
                offset <- rep(0, NROW(newx))
            }
        }
        link <- offset + cbind(1, newx) %*% coef
    }

    link <- as.matrix(link)

    if(type == "link"){
        link
    } else {
        ## Returning response
        switch(object$family,
               gaussian=link,
               binomial=inv_logit(link),
               poisson=exp(link),
			   cox = link)
    }
}

do_group_threshold <- function(x, groups){
    for(g in unique(groups)){
        g_ix <- (g == groups)
        x[g_ix] <- x[g_ix] * (abs(x[g_ix]) == max(abs(x[g_ix])))
    }
    x
}
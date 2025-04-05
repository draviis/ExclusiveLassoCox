#' CV for the Exclusive Lasso
#'
#' @rdname cv.exclusive_lasso
#' @export
#' @importFrom foreach foreach %do% %dopar%
#' @importFrom stats sd
#' @param X The matrix of predictors (\eqn{X \in \R^{n \times p}}{X})
#' @param y The response vector (\eqn{y})
#' @param groups An integer vector of length \eqn{p} indicating group membership.
#'     (Cf. the \code{index} argument of \code{\link[grplasso]{grpreg}})
#' @param ... Additional arguments passed to \code{\link{exclusive_lasso}}.
#' @param type.measure The loss function to be used for cross-validation.
#' @param nfolds The number of folds (\eqn{K}) to be used for K-fold CV
#' @param parallel Should CV run in parallel? If a parallel back-end for the
#'    \code{foreach} package is registered, it will be used. See the
#'    \code{foreach} documentation for details of different backends.
#' @param family The GLM response type. (Cf. the \code{family} argument of
#'               \code{\link[stats]{glm}})
#' @param weights Weights applied to individual
#'     observations. If not supplied, all observations will be equally
#'     weighted. Will be re-scaled to sum to \eqn{n} if
#'     necessary. (Cf. the \code{weight} argument of
#'     \code{\link[stats]{lm}})
#' @param offset A vector of length \eqn{n} included in the linear
#'     predictor.
#' @details As discussed in Appendix F of Campbell and Allen [1], cross-validation
#'    can be quite unstable for exclusive lasso problems. Model selection by BIC
#'    or EBIC tends to perform better in practice.
#' @references
#' Campbell, Frederick and Genevera I. Allen. "Within Group Variable Selection
#'     with the Exclusive Lasso." Electronic Journal of Statistics 11(2),
#'     pp.4220-4257. 2017. \doi{10.1214/EJS-1317}
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
#' exfit_cv <- cv.exclusive_lasso(X, y, groups, nfolds=5)
#' print(exfit_cv)
#' plot(exfit_cv)
#'
#' # coef() and predict() work just like
#' # corresponding methods for exclusive_lasso()
#' # but can also specify lmd="lmd.min" or "lmd.1se"
#' coef(exfit_cv, lmd="lmd.1se")



cv.exclusive_lasso <- function(X, y, groups, ...,
                                family = c("gaussian", "binomial", "poisson", "cox"),
                                offset = rep(0, NROW(X)),
                                weights = rep(1, NROW(X)),
                                type.measure = c("mse", "deviance", "class", "auc", "mae", "partial_likelihood","cindex"),
                                nfolds = 10,
                                parallel = FALSE) {
  
  tic <- Sys.time()
  
  family       <- match.arg(family)
  type.measure <- match.arg(type.measure)
  

  #stopifnot(nrow(X) > 0, ncol(X) > 0)  # Ensure X has dimensions
    # Validate response and family compatibility
  if ((family != "binomial") && (type.measure %in% c("auc", "class"))) {
    stop("Loss type ", sQuote(type.measure), " only defined for family = \"binomial.\"")
  }
  if ((type.measure %in% c("auc", "class")) && (!all(y %in% c(0, 1)))) {
    stop("Loss type ", sQuote(type.measure), " is only defined for 0/1 responses.")
  }
  
  if (family == "cox" && !type.measure %in% c("partial_likelihood", "cindex")) {
    stop("For Cox models, type.measure must be 'partial_likelihood' or 'cindex'.")
  }

   
  
  # Adjust weights if necessary
  if (sum(weights) != NROW(X)) {
    weights <- weights * NROW(X) / sum(weights)
    warning(sQuote("sum(weights)"), " is not equal to ", sQuote("NROW(X)."), " Renormalizing...")
  }
  
  # Fit the exclusive lasso model on full data
  fit <- exclusive_lasso(X = X, y = y,
                         groups = groups, family = family,
                         offset = offset, weights = weights, ...)
  
  lmd <- fit$lmd
  fold_ids <- split(sample(NROW(X)), rep(1:nfolds, length.out = NROW(X)))
  #fold_ids <- fold_ids[sapply(fold_ids, length) > 0]  # Remove empty folds
  `%my.do%` <- if (parallel) `%dopar%` else `%do%`
  i <- NULL
  # cox_loss <- function(train_fit, test_data, test_y, Xf, yf, lambda_index) {
    # #Extract coefficients for current lambda
    # coeff <- train_fit$coef[, lambda_index]
    
    # #Compute linear predictor for test and full data
    # lp_test <- as.numeric(test_data %*% coeff)
    # lp_full <- as.numeric(Xf %*% coeff)
    
    # #Compute partial likelihood for the test data
    # risk_set <- sapply(1:nrow(test_data), function(i) {
      # sum(exp(lp_test[test_data[,1][i] >= test_data[,1][i]]))
    # })
    # test_likelihood <- sum(lp_test * test_y[, 2] - log(risk_set))
    
    # #Compute partial likelihood for the full data
    # risk_set_full <- sapply(1:nrow(Xf), function(i) {
      # sum(exp(lp_full[Xf[,1] >= Xf[,1][i]]))
    # })
    # full_likelihood <- sum(lp_full * yf[, 2] - log(risk_set_full))
    
    # return(-2 * (full_likelihood - test_likelihood))  # CVE for this fold
  # }
  
  

  loss_func <- switch(type.measure,
                      mse = function(test_true, test_pred, w) weighted.mean((test_true - test_pred)^2, w),
                      mae = function(test_true, test_pred, w) weighted.mean(abs(test_true - test_pred), w),
                      class = function(test_true, test_pred, w) weighted.mean(round(test_pred) == test_true, w),
                      deviance = function(test_true, test_pred, w) weighted.mean(deviance_loss(test_true, test_pred, family), w),
                      partial_likelihood = function(train_fit, test_data, test_y, X_full, y_full, lambda_index) {
                        coeff <- train_fit$coef[, lambda_index]
                        lp_test <- as.numeric(test_data %*% coeff)
                        lp_full <- as.numeric(X_full %*% coeff)
                        
                        # Risk sets
                        risk_set_test <- sapply(1:nrow(test_data), function(i) {
                          sum(exp(lp_test[test_data[, 1] >= test_data[i, 1]]))
                        })
                        risk_set_full <- sapply(1:nrow(X_full), function(i) {
                          sum(exp(lp_full[X_full[, 1] >= X_full[i, 1]]))
                        })
                        
                        # Log-partial likelihood difference
                        test_likelihood <- sum(lp_test * test_y[, 2] - log(risk_set_test))
                        full_likelihood <- sum(lp_full * y_full[, 2] - log(risk_set_full))
                        -2 * (full_likelihood - test_likelihood)
                      },
                      cindex = function(train_fit, test_data, test_y, lambda_index) {
                        coeff <- train_fit$coef[, lambda_index]
                        lp_test <- as.numeric(test_data %*% coeff)
                        
                        # Use Cindex from survcomp package
                        Cindex_value <- glmnet::Cindex(pred = lp_test, y = test_y)
                        return(1 - Cindex_value)  # Since lower loss is better
                      },
                      stop(sQuote(type.measure), " loss has not yet been implemented."))



  cv_err <- foreach(i=1:nfolds, .inorder=FALSE,
                    .packages = c("ExclusiveLasso", "Matrix", "survival")) %my.do% {
    
            X_tr <- X[-fold_ids[[i]], ];           X_te <- X[fold_ids[[i]], ]
            y_tr <- y[-fold_ids[[i]]];             y_te <- y[fold_ids[[i]]]
            offset_tr <- offset[-fold_ids[[i]]];   offset_te <- offset[fold_ids[[i]]]
            weights_tr <- weights[-fold_ids[[i]]]; weights_te <- weights[fold_ids[[i]]]

            my_fit <- exclusive_lasso(X = X_tr, y = y_tr, groups = groups, lmd=lmd,
                                      ..., family = family, offset = offset_tr,
                                      weights = weights_tr)

            if (family == "cox") {
                coefs <- my_fit$coef
                sapply(1:ncol(coefs), function(j) {
                    if (type.measure == "cindex") {
                        loss_func(my_fit, X_te, y_te, lambda_index = j)
                    } else {
                        loss_func(my_fit, X_te, y_te, X, y, lambda_index = j)
                    }
                })
            } else {
                apply(predict(my_fit, newx = X_te, offset = offset_te), 2,
                      function(y_hat) loss_func(y_te, y_hat, weights_te))
            }
  }

  cv_err <- do.call(cbind, cv_err)
  cv_res <- apply(cv_err, 1,
                  function(x){
                    m <- mean(x)
                    se <- sd(x) / (length(x) - 1)
                    up <- m + se
                    lo <- m - se
                    c(m, se, up, lo)
                  })
  
  min_ix <- which.min(cv_res[1,])
  lmd.min <- lmd[min_ix]
  
  lmd_min_plus_1se <- cv_res[3, min_ix]
  oneSE_ix <- max(which(cv_res[1,] <= lmd_min_plus_1se))
  lmd.1se <- lmd[oneSE_ix]
  
    r <- list(fit=fit,
              lmd=lmd,
              cvm=cv_res[1,],
              cvsd=cv_res[2,],
              cvup=cv_res[3,],
              cvlo=cv_res[4,],
              lmd.min=lmd.min,
              lmd.1se=lmd.1se,
              name=type.measure,
              time=Sys.time() - tic)

    class(r) <- c("ExclusiveLassoFit_cv", class(r))

    r
}


#' @noRd
#' Deviance Loss for CV -- we define this here for readability, but it's only used
#' one small place inside CV
deviance_loss <- function(y, mu, family = c("gaussian", "binomial", "poisson")){
  family <- match.arg(family)
  
  switch(family,
         gaussian = (y - mu)^2,
         binomial = y * mu - log(1 + exp(mu)),
         poisson  = y * log(mu) - mu)
}

#' @export
print.ExclusiveLassoFit_cv <- function(x, ...){
  cat("Exclusive Lasso CV", "\n")
  cat("------------------", "\n")
  cat("\n")
  cat("lmd (Min Rule):", x$lmd.min, "\n")
  cat("lmd (1SE Rule):", x$lmd.1se, "\n")
  cat("\n")
  cat("Loss Function:", x$name, "\n")
  cat("\n")
  cat("Time: ", sprintf("%2.3f %s", x$time, attr(x$time, "units")), "\n")
  cat("\n")
  cat("Full Data Fit", "\n")
  cat("-------------", "\n")
  print(x$fit, indent=2)
  
  invisible(x)
}

#' @export
#' @importFrom stats predict
predict.ExclusiveLassoFit_cv <- function(object, ...){
  dots <- list(...)
  if("s" %in% names(dots)){
    s <- dots$s
    if(s == "lmd.min"){
      s <- object$lmd.min
    }
    if(s == "lmd.1se"){
      s <- object$lmd.1se
    }
    dots$s <- s
  }
  if("lmd" %in% names(dots)){
    lmd <- dots$lmd
    if(lmd == "lmd.min"){
      lmd <- object$lmd.min
    }
    if(lmd == "lmd.1se"){
      lmd  <- object$lmd.1se
    }
    dots$lmd  <- lmd
  }
  
  do.call(predict, c(list(object$fit), dots))
}


#' @export
#' @importFrom stats coef
coef.ExclusiveLassoFit_cv <- function(object, ...){
  dots <- list(...)
  if("s" %in% names(dots)){
    s <- dots$s
    if(s == "lmd.min"){
      s <- object$lmd.min
    }
    if(s == "lmd.1se"){
      s <- object$lmd.1se
    }
    dots$s <- s
  }
  if("lmd" %in% names(dots)){
    lmd <- dots$lmd
    if(lmd == "lmd.min"){
      lmd <- object$lmd.min
    }
    if(lmd == "lmd.1se"){
      lmd  <- object$lmd.1se
    }
    dots$lmd  <- lmd
  }
  
  do.call(coef, c(list(object$fit), dots))
}

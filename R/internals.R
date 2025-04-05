eye <- function(n) diag(1, nrow=n, ncol=n)

is_scaled <- function(X){
    ((!is.null(attr(X, "scaled:scale",  exact=TRUE))) &
         (!is.null(attr(X, "scaled:center", exact=TRUE))))
}

unscale_matrix <- function(X,
                           scale=attr(X, "scaled:scale", TRUE),
                           center=attr(X, "scaled:center", TRUE)){
    n <- NROW(X)
    p <- NCOL(X)

    X * matrix(scale, n, p, byrow=TRUE) + matrix(center, n, p, byrow=TRUE)
}


capitalize_string <- function(x){
    paste0(toupper(substring(x, 1, 1)), substring(x, 2))
}

clamp <- function(x, range){
    pmin(pmax(x, min(range)), max(range))
}


lmd_interp <- function(x, old_lmd, new_lmd){
    new_lmd <- clamp(new_lmd, range(old_lmd))

    lb <- vapply(new_lmd, function(x) max(which(old_lmd <= x)), numeric(1))
    ub <- vapply(new_lmd, function(x) min(which(old_lmd >= x)), numeric(1))

    lb <- clamp(lb, c(1, length(old_lmd)))
    ub <- clamp(ub, c(1, length(old_lmd)))

    frac <- (new_lmd - old_lmd[lb]) / (old_lmd[ub] - old_lmd[lb])

    frac[lb == ub] <- 1


    frac * x[, lb, drop=FALSE] + (1-frac) * x[, ub, drop=FALSE]
}


#' @importFrom stats plogis
inv_logit <- function(x) plogis(x)

icat <- function(..., indent=0){
    if(indent){
        cat(rep(" ", indent), sep="")
    }
    cat(...)
}

logspace <- function(x, y, length.out){
    exp(seq(log(x), log(y), length.out=length.out))
}

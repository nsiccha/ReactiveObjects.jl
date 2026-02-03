@reactive dual_averaging_state(init; target=.8, regularization_scale=.05, relaxation_exponent=.75, offset=10) = begin 
    m = one(init)
    H = zero(init)
    mu = log(10) + log(init)
    log_current = mu - sqrt(m) / regularization_scale * H
    log_final = zero(init)
    current = exp(log_current)
    final = exp(log_final)
    fit!(x) = begin 
        m += 1 
        H += (target - x - H) / (m + offset)
        log_final += m^(-relaxation_exponent) * (log_current - log_final)
    end
end
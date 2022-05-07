function S = soft_threshold(gradient,tuning)

S = sign(gradient).*subplus(abs(gradient)-tuning);
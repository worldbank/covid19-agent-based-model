
# Assuming the mean risk of severe disesae is 0.2
risk_severe_covid <- data.frame(age = seq(0, 100, 20),
                                risk = seq(0,0.4, length.out = 6))

risk_severe_covid

# get the mean risk across ages
mean_risk <- mean(risk_severe_covid$risk) #0.2

# If the predicted mean risk of severe disease is 0.25
risk_severe_covid$risk * 0.25/mean_risk

# If the predicted mean risk of severe disease is 0.18
risk_severe_covid$risk * 0.18/mean_risk

---
title: "A Simple Example to Gain Insight into Bayesian Inference"
date: "2025-04-23"
description: "A practical, professional walkthrough of Bayesian inference in structural engineering, with a real-world scenario and step-by-step calculation."
category: "statistics"
tags: ["statistics", "bayesian inference", "probability", "engineering", "case study"]
author: "M. Talebi"
readTime: "8 min read"
---

# Scenario

You are a structural engineer assessing the condition of a building structure. You suspect that the building might be damaged due to external factors. To investigate, you conduct a vibration test on the building. You want to calculate the probability that the building is actually damaged given that the vibration test result indicates damage.

# Given Information

- **Prior knowledge:** Historically, only 5% of buildings in your region have experienced significant damage.
- **Vibration test accuracy:** The vibration test has an accuracy of 90%, meaning it correctly identifies damage 90% of the time.

# Perspectives

1. **Percy Pessimist:** Percy believes that if the test indicates damage, it's highly likely that the building is damaged, even if only a small percentage of buildings historically experienced damage.
2. **Opal Optimist:** Opal considers the test's potential for errors and believes that the building might not be damaged despite the test result. She thinks the high accuracy might not be perfect.
3. **Fin Frequentist:** Fin uses data to calculate the likelihood of observing a positive test result given a damaged building. He concludes that the building is likely to be damaged due to the high test accuracy.
4. **Ben Bayesian:** Ben uses Bayes' theorem to calculate the posterior probability that the building is not highly likely to be damaged, given a positive test result. He combines test accuracy with prior knowledge to do so.

# Ben's Bayesian Approach

- Ben uses Bayes' theorem to calculate the posterior probability of the building being damaged given a positive test result:

---

> 💡 **Recap**
> 
> $ \text{Posterior} = \frac{\text{Likelihood} \cdot \text{Prior}}{\text{Evidence}} $
> 
> Evidence is also called *Marginal Likelihood*
> 
> $ P(\text{Building Damaged} \mid \text{Positive Test Result}) = \\
> \frac{P(\text{Positive Test Result} \mid \text{Building Damaged}) \cdot P(\text{Building Damaged})}{P(\text{Positive Test Result})} $
> 
> $$
> \begin{aligned}
>   P(\text{Positive Test Result}) = & P(\text{Building Damaged}) \cdot P(\text{Positive Test Result} \mid \text{Building Damaged}) \\
>   & + P(\text{Building Undamaged}) \cdot P(\text{Positive Test Result} \mid \text{Building Undamaged})
> \end{aligned}
> $$

---

$ P(\text{Building Damaged} | \text{Positive Test Result}) = \frac{0.9 \cdot 0.05} {0.15}=0.30 $

$ P(\text{Positive Test Result})=0.05\cdot0.90+0.95\cdot0.1=0.15 $

From Ben's perspective, considering prior knowledge about buildings in this region, it is not highly probable to say that the building is damaged.

![Bayesian Posterior Illustration](https://mtalebi.com/wp-content/uploads/2025/04/image-3.png)

If we have no prior knowledge about the buildings in the region, or if we ignore the prior knowledge, we could assume a uniform probability (0.5) of having damaged or undamaged buildings in the region (Fin's perspective).

$ P(\text{Positive Test Result})=0.5\cdot0.90+0.5\cdot0.1=0.5 $

$ P(\text{Building Damaged} | \text{Positive Test Result}) = \frac{P(\text{Positive Test Result} | \text{Building Damaged}) \cdot 0.5} {0.5}$

$=P(\text{Positive Test Result} | \text{Building Damaged})=0.9$

That is why Fin believes that the building is likely to be damaged due to the 90% accuracy of the test. 
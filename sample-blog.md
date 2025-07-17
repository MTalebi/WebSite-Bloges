---
title: "Sample Blog: Full Markdown & Features Demo"
date: "2024-06-01"
description: "A comprehensive test post to demonstrate all blog features: tables, Python code, math, images, callouts, and more."
author: "Test User"
tags: ["demo", "python", "table", "markdown", "math", "image", "callout"]
category: "tutorial"
readTime: "4 min read"
---

# Sample Blog Post: Full Markdown & Features Demo

This sample post demonstrates all the main features supported by the blog system, including:
- Tables
- Python code blocks (with syntax highlighting, line numbering, copy button)
- Inline and block math (LaTeX)
- Images with captions
- Blockquotes and callouts
- Lists, links, and more

---

## 1. Table Example

| Name      | Role         | Score |
|-----------|--------------|-------|
| Alice     | Engineer     | 95    |
| Bob       | Scientist    | 88    |
| Charlie   | Analyst      | 92    |

---

## 2. Python Code Example

```python
def fibonacci(n):
    """Return the nth Fibonacci number."""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

for i in range(10):
    print(f"Fibonacci({i}) = {fibonacci(i)}")
```

---

## 3. Math Equations

Inline math: $E = mc^2$ is the most famous equation in physics.

Block math:

$$
\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}
$$

Numbered block math:

$$
\begin{align}
    a^2 + b^2 &= c^2 \\
    e^{i\pi} + 1 &= 0
\end{align}
$$

---

## 4. Figure/Image with Caption

![Gaussian Distribution](../assets/images/playground/gaussian-dist.png "Gaussian Distribution{width=60%}")

A plot of the standard normal (Gaussian) distribution.

---

## 5. Blockquote Callout

> 💡 **Tip:** You can copy the code above using the copy button in the top-right corner of the code block.

---

## 6. Lists and Links

- This is a bullet list item
- Another item
    - Nested item

1. Numbered list item
2. Another numbered item

[Visit the main website](../index.html)

---

## 7. Table with Caption

<caption>Team Scores</caption>
| Team   | Points |
|--------|--------|
| Red    | 10     |
| Blue   | 8      |
| Green  | 12     |

---

## 8. Inline Code

Use `pip install numpy` to install NumPy.

---

## 9. Horizontal Rule

---

## 10. End of Template

Feel free to use this as a template for your own posts, or add/remove sections to test specific features! 

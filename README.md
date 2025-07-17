# Blog Repository – Writing Posts in Markdown

This repository contains the markdown source files for blog posts published on the main website. Follow these guidelines to ensure your posts render beautifully and professionally.

---

## How to Write a Blog Post

### 1. Create a Markdown File
- Name your file descriptively, e.g., `my-first-post.md`.

### 2. Add YAML Front Matter
At the top of your file, include:
```markdown
---
title: "Your Post Title"
date: "YYYY-MM-DD"
description: "A short summary of the post."
author: "Your Name"
tags: ["tag1", "tag2", ...]
category: "tutorial"  # or "research", "update", etc.
readTime: "5 min read"
---
```

### 3. Write Your Content in Markdown
You can use all standard markdown features, plus:

#### Table Captions
To add a caption to a table, place a line starting with `^caption:` immediately before the table. The caption will be rendered as a left-aligned, numbered caption. Only "Table N." is bold.
```markdown
^caption: Monthly Statistics
| Month | Value |
|-------|-------|
| Jan   | 100   |
| Feb   | 120   |
```

#### Code Blocks
Use triple backticks and specify the language for syntax highlighting. Example:
```markdown
```python
def hello():
    print("Hello, world!")
```
```
- Code blocks have line numbers and a copy button.

#### Math (LaTeX)
- Inline: `$E = mc^2$`
- Block:
  ```markdown
  $$
  \int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}
  $$
  ```
- Numbered block (align):
  ```markdown
  $$
  \begin{align}
      a^2 + b^2 &= c^2 \\
      e^{i\pi} + 1 &= 0
  \end{align}
  $$
  ```

#### Images with Captions and Width
```markdown
![Gaussian Distribution](../assets/images/playground/gaussian-dist.png "Gaussian Distribution{width=60%}")
```
- The caption is taken from the image title.
- Width can be set with `{width=XX%}` in the title.

#### Callouts (Tips)
```markdown
> 💡 **Tip:** You can copy the code above using the copy button.
```
- Renders as a styled callout box.

#### Lists and Links
- Bullet list:
  ```markdown
  - Item 1
  - Item 2
  ```
- Numbered list:
  ```markdown
  1. First
  2. Second
  ```
- Link: `[My Website](../index.html)`

#### Inline Code
Use backticks: `` `pip install numpy` ``

#### Horizontal Rule
Use `---` for a horizontal line.

---

## Adding Your Post to the Blog
1. Commit and push your markdown file to this repo.
2. Edit `blog-index.json` and add an entry for your post:
   ```json
   [
     {
       "filename": "https://raw.githubusercontent.com/YourUsername/YourBlogRepo/main/my-first-post.md",
       "order": 1
     },
     ...
   ]
   ```
3. Commit and push the updated `blog-index.json`.
4. Your post will appear on the main website automatically.

---

## Best Practices
- Use clear, descriptive titles and summaries.
- Add relevant tags and categories for discoverability.
- Use math, code, images, and tables for technical clarity.
- Keep posts professional and well-structured.
- Preview your post on GitHub to check markdown rendering.
- Use the `^caption:` feature for professional table captions.

---

Happy blogging!

---
name: monthly-readings
description: Use when generating a monthly "reading" post in _posts with a list of provided URL links. Fetches contents, generates concise summaries, and appends formatted entries to markdown posts.
---

# Monthly Readings Article Generator

This skill guides GitHub Copilot through constructing monthly "reading" articles in the `_posts/` folder from a provided list of URLs.

## Task Workflow

When the user asks to write a reading article and provides a list of URLs, execute the following steps sequentially for **every link provided**:

### Step 1: Fetch Article Content
1. Access and read the target URL to extract the article's core content and main title.
2. If unable to fetch full web content directly, use available search/fetch tools or ask the user for context regarding the link.

### Step 2: Generate Summary
1. Analyze the main key takeaways and highlights of the fetched article.
2. Draft a concise summary under **500 characters** in length.
3. Keep the tone insightful, clear, and direct.

### Step 3: Format Output Block
Format each processed link strictly according to this template:

```markdown
**Title: <Article Title>**

[Link](<URL>)

Summary: <Article 500 Summary characters under>

---
```

### Step 4: Write to Target Markdown Post
Identify the active or target monthly reading post in _posts/ (e.g., _posts/YYYY-MM-DD-readings-month-year.md). If none exists, ask the user or create a new Jekyll/Markdown blog post header.

Append each formatted link block into the post body one by one until all provided links are completed.

### Formatting Guidelines & Constraints
- **Strict Character Limit**: Summaries MUST remain under 500 characters.

- **Title Accuracy**: Extract the exact or clean main headline from the article.

- **Separator**: Every link block MUST end with a horizontal rule (---).

- **Batching**: Process all URLs provided in the prompt in sequential order without skipping any link.


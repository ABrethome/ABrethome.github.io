---
title: "Applying the lessons from the 'AI-for-Science' hype to the corporate world"
date: 2025-07-27
permalink: /posts/2025/07/ai-for-science-hype/
tags:
  - AI
  - Hype
---

**Disclaimer:** The views expressed in this post are solely my own and do not represent those of my employer.
{: .notice}

Like many researchers, I've felt the irresistible pull of the AI promise. In academia, that translates into a pressure to "surf the hype", believing that the fastest way to career growth is to attach a neural network to a tough problem and hope for magic.

In the corporate world, the same fever takes a different shape. Here, it's the "Shiny Hammer" syndrome: teams are told to "use AI" first, and only later try to figure out what problem they're actually solving. The result? Projects that look dazzling on a slide deck but deliver little more than disappointment and wasted resources.

I learned this lesson the hard way. I was genuinely excited about using Physics-Informed Neural Networks (PINNs) for chemistry in a personal project. That excitement fizzled out when I stumbled upon [Nick McGreivy's brutally honest critique of the AI-for-science hype](https://www.understandingai.org/p/i-got-fooled-by-ai-for-science-hypeheres?r=2zm2nw&utm_campaign=post&utm_medium=email&triedRedirect=true). His post perfectly captured my own journey: from the thrill of reading an elegant paper to the slow, painful realization that the approach was riddled with silent flaws and overblown claims. Sometimes, published work is just a monument to a single niche success, with no self-criticism or analysis of its failure modes.

What really hit me was how the pitfalls Nick described aren't just academic; they're everywhere, including the corporate world where research is alive and well.

So, how did we end up here? What keeps this cycle of AI hype spinning out of control?

## AI-for-Science pitfalls in academia

Nick McGreivy boils down the AI-for-science pitfalls into five sharp points. The perceived success of AI in science is often a mirage, shaped by academic and industry culture; not an accurate reflection of its true utility:

1. **Scientific literature can be unreliable due to survivorship and reporting bias**: Negative results rarely see the light of day. Researchers are rewarded for publishing only their wins, so the literature is skewed toward impressive outcomes. Failures-like PINNs floundering on certain PDEs, as I saw firsthand; are swept under the rug.
2. **AI's superiority is often an illusion, thanks to weak baselines and unfair comparisons**: The most jaw-dropping claims ("orders of magnitude faster!") usually come from comparing AI to outdated, poorly-performing methods. Stack AI up against a modern, state-of-the-art approach, and the magic often vanishes.
3. **Personal incentives drive AI adoption, sometimes with glaring conflicts of interest**: Scientists chase AI not always for the good of science, but for personal gain: higher salaries, prestige, bigger grants, more citations. This leads to a "hammer in search of a nail" mentality, where AI is assumed to be the answer before the question is even asked.
4. **Overly optimistic results are everywhere, thanks to cherry-picking, misreporting, and data leakage**: Researchers evaluating their own models have a built-in conflict of interest. Methodological errors like data leakage and cherry-picking successes while ignoring failures are rampant.
5. **AI models can be brittle and lack rigor**: Even celebrated methods like PINNs turn out to be finicky and unreliable in practice. They lack the theoretical guarantees and robust validation that traditional scientific methods offer.

These five points struck a chord with me because I've seen them up close. The temptation is real! Most of us don't fall into these traps on purpose - it's just human nature. I know I'll stumble into one of these pitfalls again, and I hope this article will serve as a reminder to myself.

But these lessons aren't just for academia. The same patterns play out in the corporate world, and the parallels are uncanny.

## The corporate mirror: AI pitfalls at work

Recently, I've grown frustrated with one of our vendors. The AI-for-science pitfalls fit the corporate world like a glove. In business, the drive for profit, market share, and investor confidence replaces academic prestige - but drivers are similar. The "oversell" becomes a calculated strategy.

For users like me, it's maddening. The result? Wasted effort, wasted money, and a lot of broken promises.

Let's revisit four of those five pitfalls, now dressed in business attire:

1. **Survivorship bias/reporting bias-only marketing the success story**: Vendors showcase spectacular case studies ("Customer X achieved a 20% efficiency gain using AI!") while quietly ignoring the common failures and brittleness most users experience. For customers, this means managing unrealistic expectations and investing in tools that were never fit-for-purpose. We pour time, money, and talent into a solution, only to watch it crumble on our messy, real-world data.
2. **Weak baselines/unfair comparisons-using "legacy" benchmarks**: Vendors compare their AI tool to the customer's old, inefficient system or a non-optimized alternative - not to a strong, modern method."Our AI tool is 25% more efficient!* *than our v1 tool." Yes, I've actually seen this from an AI-first company. The result? Customers buy expensive solutions that are only marginally better than cheaper, non-AI options. AI is powerful, but not always necessary, and the cost-benefit analysis is a must-have.
3. **Conflict of interest**: The vendor's sales team, product evangelists, and internal researchers are the loudest voices touting success. Their financial incentives are tied to adoption, making them unreliable narrators. That's a given. But customers can get trapped in a "citation bubble," where everyone at a conference is hyping the same tech ; sometimes without having finished the journey themselves. In the end, there are people looking like reliable narrators who don't, and we invest in tools that may soon be abandoned, unproven, or superseded, leading to sunk costs and dead-end dependencies.
4. **Lack of robustness-generalization failure**: The AI tool dazzles on tidy, demo data but falls apart on the customer's complex, "dirty" internal data. For me, this means operational instability and risk. The tool is too finicky and unreliable for mission-critical tasks, requiring human oversight, costly retraining, and introducing errors-defeating the whole point of "acceleration". Even a lot of pre-work before adopting a tool is sometimes not enough.

## Breaking the cycle?

I've lived these pitfalls, both in academia and at work. The AI hype cycle is seductive, but it's also exhausting. The real challenge is to stay curious, skeptical, and honest ; especially when the next shiny tool comes knocking.

So, what about you? Have you found yourself caught in the AI hype like me, either as a researcher or a customer? What lessons have you learned, and how do you spot the warning signs before it's too late?

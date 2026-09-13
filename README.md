# Agentic Multimodal Product Decision System

This project demonstrates an LLM-based decision agent that combines personalized retrieval, request-specific catalog search, product metadata, and review evidence to answer product recommendation and preference-summary requests.

The recommendation process is driven by local multimodal representations and retrieval utilities that provide structured evidence, while the agent interprets the user's current intent and makes the final decision, with the language model serving as an integrated component of this pipeline.

The personalized recommender is intentionally lightweight. Its purpose is to provide useful candidate products to the agent rather than maximize recommendation benchmark performance. A high vector-similarity score is therefore treated as one decision signal, not as an automatic recommendation.

## What the agent does

The system supports two high-level behaviors:

- **Preference summary**: explain what a user historically liked, disliked, used, rated, or reviewed.
- **Recommendation**: combine the user's historical preference signal with the current natural-language request, product facts, and review evidence to select products.

For recommendation requests, the system can draw on four complementary capabilities:

1. **Personalized retrieval** from multimodal user and product vectors.
2. **Request-specific catalog search** for explicit needs such as product type, brand, ingredient, price, scent, texture, hair condition, skin condition, or color.
3. **Product information lookup** for factual product attributes.
4. **Review evidence lookup** using representative favorable and unfavorable reviews for finalist products.

The final answer is produced by integrating these signals, with vector similarity serving as one component of the overall decision process.

## Architecture

### 1. Multimodal product representation

```text
                         PRODUCT REPRESENTATION

Product image ---- DINOv2 Base ---- z_image ----+
                                                +-- weighted concat --> product_vec
Product title ---- BGE Base v1.5 --- z_title ----+
```

The current encoders are:

- Image encoder: `facebook/dinov2-base`
- Title encoder: `BAAI/bge-base-en-v1.5`
- Fusion: normalized weighted concatenation
- Default weights: 40% image and 60% title

The two embedding spaces are not added element by element. Each modality is L2-normalized first, then concatenated with square-root weights:

```text
z_image = normalize(DINOv2(image))
z_title = normalize(BGE(title))

product_vec = normalize(concat(sqrt(0.4) * z_image,
                               sqrt(0.6) * z_title))
```

Square-root scaling makes the final dot product contribute approximately 40% image similarity and 60% title similarity.

### 2. Lightweight user preference representation

A user vector is constructed from the multimodal vectors of products the user rated:

```text
rating 1 --> -2
rating 2 --> -1
rating 3 -->  0
rating 4 --> +1
rating 5 --> +2

user_vec = normalize(sum((rating - 3) * product_vec))
```

This creates a content-based preference direction in the same space as the product vectors. Products rated positively pull the profile toward their multimodal representations, while negatively rated products push it in the opposite direction.

For repeated user-product interactions, only the most recent rating is used. A user with only neutral ratings, or perfectly cancelled positive and negative signals, receives a zero vector and cannot use vector ranking.

### 3. Agent tools and decision flow

```text
                              User request
                                   |
                                   v
                           Request interpretation
                                   |
                    +--------------+--------------+
                    |                             |
                    v                             v
           Preference summary                Recommendation
                    |                             |
                    |                 +-----------+-----------+
                    |                 |           |           |
                    |                 v           v           v
                    |          Personalized   Catalog     Product facts
                    |            retrieval     search
                    |                 |           |
                    |                 +-----+-----+
                    |                       |
                    |                       v
                    |                Candidate products
                    |                       |
                    |                       v
                    |                Review evidence
                    |                       |
                    +-----------------------+
                                            |
                                            v
                                   GPT decision layer
                                            |
                                            v
                                  Title + Reason + Caution
```

The important separation is:

```text
retrieval tools --> gather candidates and evidence
LLM agent       --> interpret the request and make the final decision
```

A product with the highest personalized vector score does not automatically win. A lower-ranked product can be selected when it better satisfies the user's current request or has stronger supporting review evidence.

### Current V1 orchestration

The current implementation uses a **hybrid orchestration design** rather than an unconstrained autonomous loop.

For recommendation requests:

1. Retrieve up to 50 unseen products with `user_vec x product_vec` when a valid user ID is available.
2. Search for up to 30 products that match the current request.
3. Merge candidate sets by `parent_asin` and remove duplicates.
4. Apply explicit request constraints before personalized similarity scores.
5. Read product facts plus representative favorable and unfavorable reviews for finalists.
6. Let the GPT decision layer return one to five products.

This design keeps candidate generation deterministic and inspectable while still letting the language model reason over user intent, conflicting signals, and review evidence.

A final product does not need to come from the personalized Top-50. A request-specific search result can win when it satisfies the current need better.

The user's own review text is optional context. The agent may use a concrete preference expressed in a review, such as scent, texture, irritation, durability, skin type, or hair type. It does not force a connection when that text is vague or irrelevant, and it does not expose internal history-processing language in the final answer.

Review vectors are not currently precomputed. BGE is used for product titles; review evidence is selected from reviews attached to finalist products. This keeps the documented architecture aligned with the implementation.

## Request routing

The agent supports two request types:

- `preference_summary`: questions about what the user previously liked, disliked, used, rated, or reviewed. The system reads only that user's interactions and answers directly. It does not retrieve candidates or recommend new products.
- `recommendation`: requests for products to buy, try, compare, or use for a current need. The system gathers candidate products and evidence, then returns `Title`, `Reason`, and `Caution`.

For example:

```text
What type did I use to like?
```

is a preference-summary request, while:

```text
Recommend a product similar to the types I used to like.
```

is a recommendation request.

## Repository files

```text
github_product_agent/
  product_recommendation_agent.py  Request routing, local retrieval tools, and GPT decision layer
  build_product_embeddings.py      DINOv2 + BGE product embedding pipeline
  build_user_embeddings.py         Rating-weighted user embedding export
  requirements.txt                 Python dependencies
  .env.example                     API key template
  .gitignore                       Excludes secrets and large artifacts
  README.md                        Architecture and usage
```

## Required local data

Large datasets, images, and tensor files are intentionally excluded from Git. During local development, place `dataset/` and `image/` either inside this directory or beside it:

```text
dataset/
  All_Beauty_2021_2023.jsonl
  meta_All_Beauty.jsonl

image/
  <parent_asin>.jpg
```

Generated tensors are also stored under `dataset/` and remain excluded from Git.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
cp .env.example .env
```

Set the API key in `.env`:

```text
OPENAI_API_KEY=your_api_key_here
```

The embedding scripts do not require an OpenAI API key. Only the decision agent calls the OpenAI API.

## Step 1: Build product embeddings

```bash
python build_product_embeddings.py
```

Default output:

```text
dataset/All_Beauty_2021_2023_users_5plus_dinov2_bge_embeddings.pt
```

The output contains `parent_asins`, `image_vec`, `text_vec`, `product_vec`, encoder names, dimensions, fusion weights, processing statistics, and skipped-image errors.

The image and title models are loaded sequentially, not simultaneously. Default batch sizes are selected for an 8 GB RTX 4060:

```bash
python build_product_embeddings.py \
  --image-batch-size 32 \
  --text-batch-size 256
```

If CUDA runs out of memory, reduce `--image-batch-size` to 16 or 8. Product embeddings only need to be rebuilt when the product catalog, images, encoders, or fusion weights change.

To change modality weights:

```bash
python build_product_embeddings.py --image-weight 0.4
```

The title weight is automatically calculated as `1 - image_weight`.

## Step 2: Build user embeddings

```bash
python build_user_embeddings.py
```

Default output:

```text
dataset/All_Beauty_2021_2023_users_5plus_user_embeddings.pt
```

The output contains `user_ids`, `user_vec`, `has_preference_signal`, `profile_strength`, `non_neutral_counts`, the rating formula, product model information, and processing statistics.

This offline export is useful for analysis or bulk retrieval. The agent computes the same rating-weighted profile directly for the requested user so it can use the current local review file.

## Step 3: Run the decision agent

```bash
python product_recommendation_agent.py \
  --user-id AFSKPY37N3C43SOI5IEXEK5JSIYA \
  --request "Recommend a lightly scented product for dry, curly hair."
```

The user ID can also appear in the request text. If the ID exists in the local active-user data, the script detects it automatically:

```bash
python product_recommendation_agent.py \
  --request "I am user AFSKPY37N3C43SOI5IEXEK5JSIYA. Recommend a hair product."
```

The visible recommendation contains exactly three fields per product:

```text
Top1
Title: Product title
Reason: Concise product-fit explanation
Caution: Relevant limitation or safety note
```

Internal product identifiers, vector scores, retrieval details, and processing steps are not included in the visible recommendation.

## Examples

### Example 1: Personalized recommendation with a specific hair-care request

Command:

```bash
python product_recommendation_agent.py \
  --user-id AFSKPY37N3C43SOI5IEXEK5JSIYA \
  --request "Recommend a lightly scented product for dry, curly hair."
```

Example answer:

```text
Top1
Title: KHADI ROSE REPAIR Ayurvedic shampoo, 100% natural, silicone & sulfate-free, deep conditioning & regeneration for structurally damaged, dry hair & split ends, vegan hair care, organic beauty, 6.7oz
Reason: Floral-scented, hydrating Ayurvedic shampoo that repairs dry, damaged hair.
Caution: Fragrance may be strong for fragrance-sensitive users.

Top2
Title: Garnier Whole Blends Softening Shampoo Bar for Fine to Normal Hair, Oat Delicacy, 2 Oz, 1 Count (Packaging May Vary)
Reason: Light honey-scented shampoo bar with plastic-free packaging; gentle cleansing for hair.
Caution: Not ideal for very dry curly hair; some users reported frizz or small bar size.

Top3
Title: Taya Beauty Buriti Nut Intensive Repair Shampoo & Conditioner Duo - Organic Hydrating Shampoo and Conditioner for Dry Damaged Hair - Restorative Hair Care - Travel Size 2 fl oz
Reason: Hydrating duo designed to moisturize and restore dry, damaged hair, with a pleasant scent noted by users.
Caution: Small bottles; pricey; fragrance may be noticeable.
```

### Example 2: Request-specific recommendation without a user ID

Command:

```bash
python product_recommendation_agent.py \
  --request "Find me a moisturizer under €20."
```

Example answer:

```text
Top1
Title: Hyalogic HA Daily Skin Perfecting Lotion 1oz - The Premium Spa-Grade Hyaluronic Acid Facial Moisturizer to Control Oiliness and Improve Skin Texture - 1 Fl Ounce
Reason: Under €20; hydrates with hyaluronic acid and helps control oil for smoother skin.
Caution: May feel oily or not sufficiently moisturizing for very dry skin; some users reported an oily feel after application.
```

### Example 3: Preference summary without recommendations

Command:

```bash
python product_recommendation_agent.py \
  --user-id AFTLUVGQWKW6XSQ5TB6UER5Q263A \
  --request "What type did I used to like?"
```

Example answer:

```text
Based on your history, you tended to like beauty tools and spa-like skincare routines. You repeatedly rated facial massage tools highly, especially rose quartz and jade rollers and gua sha sets, indicating a preference for cooling, massage-style application. You also show strong interest in hydrating and brightening skincare products, such as Vitamin C cleansers and creams, honey masks, eye patches, and botanical serums. You seem to favor affordable, multi-pack or travel-friendly sets, and you respond well to pleasant scents, such as citrus or floral scents, in cleansers and masks. You also often used supportive accessories, such as headbands and hair ties, to keep hair away during routines.
```

## Design scope

This repository is a V1 agent demonstration rather than a recommendation-system benchmark project.

The current design intentionally keeps the recommendation model simple:

- frozen pretrained image and text encoders;
- weighted multimodal concatenation;
- rating-weighted user profiles;
- cosine-similarity personalized retrieval.

The main project focus is the decision layer that combines user history, current intent, catalog search, product facts, and review evidence.

Possible future extensions include learned user-item ranking, temporal user modeling, embedded review retrieval, uncertainty estimation, and a fully dynamic tool-calling loop.

## Git and data safety

The included `.gitignore` excludes `.env`, JSONL and compressed datasets, PyTorch tensor files, image directories, notebook caches, and Python caches. Only source code, documentation, the dependency list, and `.env.example` should be committed.

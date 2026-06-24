
" GNN Models + BPR Model (user-user similarity)
Recommend theo lịch sử mua hàng
Những user có pattern mua hàng giống bạn → họ cũng hay mua item X → gợi ý X cho bạn
"Dự đoán xác suất user sẽ mua một item cụ thể"
Input: (user_id, item_id) → Output: score 0–1
"

"" Siamese Model (purchase sequence per user)
User vừa mua item A → item B có phải là thứ họ mua tiếp theo không?" "item nào thường được mua sau item X" để recommend
Hai item hay được mua liên tiếp bởi cùng một user → vector gần nhau
Item phổ biến nhưng không liên quan → vector xa nhau
""

==============================================================================================
#### Step 1: preprocess
```
python preprocess.py --dataset hm
```

==============================================================================================
#### Step 2: extract embedding
```
python extract.py --dataset hm --feature clip

python extract.py --dataset hm --feature fashionclip
```

==============================================================================================

#### Step 3: build graph
```
python build_graph.py --dataset hm --feature clip

python build_graph.py --dataset hm --feature fashionclip
```

==============================================================================================
#### Step 4: train model
```
python train.py --dataset hm --feature clip --model lightgcn
python train.py --dataset hm --feature clip --model siamese

python train.py --dataset hm --feature fashionclip --model lightgcn
python train.py --dataset hm --feature fashionclip --model siamese
```

==============================================================================================

#### Step 5: evaluate
```
# GNN models (cần --feature)
python evaluate.py --dataset hm --feature clip   --model lightgcn
python evaluate.py --dataset hm --feature clip   --model graphsage
python evaluate.py --dataset hm --feature clip   --model ngcf


python evaluate.py --dataset hm --feature fashionclip   --model lightgcn
python evaluate.py --dataset hm --feature fashionclip   --model graphsage
python evaluate.py --dataset hm --feature fashionclip   --model ngcf


# Baseline (BPR không cần --feature)
python evaluate.py --dataset hm --model bpr

# Siamese (cần --feature)
python evaluate.py --dataset hm --feature clip   --model siamese
python evaluate.py --dataset hm --feature fashionclip   --model siamese
        
# Đánh giá tất cả một lúc
python evaluate.py --dataset hm --all

```# fashion_recommendation

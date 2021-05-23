# bash file to execute CSML

python -m csml.src \
    --train "./data/train/" \
    --label "./data/label/" \
    --output "./result/example/" \
    --model "./data/model.pkl" \
    --infer "./data/infer/" \
    --mode "back" \
    --ntrain 20000 \
    --discard 64 \
    --close 1 \
    --height 64 \
    --width 64 \


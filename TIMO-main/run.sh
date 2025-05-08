for DATA in caltech101 dtd eurosat fgvc food101 oxford_flowers oxford_pets stanford_cars ucf101 sun397 imagenet
do 
    for SHOOT in 1 2 4 8 16
    do
        for SEED in 1 2 3
        do
            CUDA_VISIBLE_DEVICES=0 python main.py --config configs/${DATA}.yaml --shot ${SHOOT} --seed ${SEED} 
        done
    done
done
from generator import botr_generator

if __name__ == "__main__":

    coco_dset = torchvision.datasets.CocoDetection(
    "val2017",
    "annotations/stuff_val2017.json", 
    transform=transforms.ToTensor())
    
    botr_gen = botr_generator(
        coco_dset, 
        # 
        dims=(128,128),
        batch_size=4,
        fill_target=0.99,
        max_step_fill=0.1,
        step_fill_jitter=0.5)
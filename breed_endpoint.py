import api
from dataset import Dataset, DATASET_CONFIG
from botr import BOTR
from config import RENDER_CONFIG_DEFAULT, GENERATOR_CONFIG_DEFAULT
from nft import Breedable_NFT, add_parents_to_creators
from metaplex import format_asset_metaplex
from utils import save_json
from os.path import join
import time

CLIENT_RENDER = {
    'matchHistograms' : False,
    'compositeType' : 'binary',
    'outputSize' : (256, 256),
    'jpeg_quality' : 100,
    'adaptiveHistogram' : False,
    'showProgress' : True,
    "batchRenderSize" : 8 }

NFT_CONFIG = {
    'inheritence_fill' : 0.75,
    'target_fill' : 0.96,
    'overstep_lim' : 1.5 }

PUBLIC_MEDIA = "/public-media-library"
SAVE_DIR = "/botr/breed-request"
FULL_SAVE_DIR = f"{PUBLIC_MEDIA}{SAVE_DIR}"
SLEEP_TIME = 1

def breed(parent1_id: str, parent2_id: str, breed_id: str, dataset: Dataset):
    parent1 = Breedable_NFT(parent1_id)
    parent2 = Breedable_NFT(parent2_id)
    botrGen = BOTR(config=GENERATOR_CONFIG_DEFAULT, registerProductInfo=False)
    botrGen = parent1.breed(parent2, botrGen, dataset, nftConfig=NFT_CONFIG)
    genItem = botrGen.generate(CLIENT_RENDER)
    img_path = join(FULL_SAVE_DIR, f"{breed_id}.png")
    meta_path = join(FULL_SAVE_DIR, f"{breed_id}.json")
    metadata = format_asset_metaplex(img_path, genItem.metadata, None)
    metadata = add_parents_to_creators(
        metadata=metadata, parent_1=parent1,
        parent_2=parent2, child_attrs=genItem.metadata["composition"])
    genItem.image.save(img_path)
    save_json(meta_path, metadata)
    print(f'=> Saving breed request assets to {img_path} {meta_path}')
    response = {
        "id" : breed_id,
        "status" : "complete",
        "previewImageUri" :  f"https://homunculi.org{FULL_SAVE_DIR}/{f'{breed_id}.png'}",
        "previewMetaUri" : f"https://homunculi.org{FULL_SAVE_DIR}/{f'{breed_id}.json'}",
        "statusMessage" : f"Saving image: {img_path} Saved json: {meta_path}" }
    return response

def handle_response(breed_request, dataset: Dataset):
    for b in breed_request:
        print("SENDING NEW BREED REQ")
        try:
            response = breed(b['parent1'], b['parent2'], b['id'], dataset)
            api.complete_breed_request(response)
        except Exception as e:
            print(e)
            api.complete_breed_request({
                "id" : b["id"],
                "status" : "failed",
                "previewImageUri" : "",
                "previewMetaUri" : "",
                "statusMessage" : str(e) })

def main():
    dataset = Dataset()
    while True:
        # try:
        breed_request = api.request_breed_status("new")
        if breed_request is not None and len(breed_request) > 0:
            print(f"=> Handing breed request {breed_request}")
            handle_response(breed_request, dataset)
        time.sleep(SLEEP_TIME)

if __name__ == "__main__":
    main()












# async def work():
#     while True:
#         print("doing shit")
#         await asyncio.sleep(SLEEP_TIME)
#         breed_request = api.request_breed_status()
#         if breed_request is not None:
#             print(f"=> Handing breed request {breed_request}")
#             handle_response(breed_request, dataset)

# loop = asyncio.get_event_loop()
# try:
#     asyncio.ensure_future(work())
#     loop.run_forever()
# except KeyboardInterrupt:
#     loop.close()
# finally:
#     print("Closing Loop")
#     loop.close()
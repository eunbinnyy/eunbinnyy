import os
import replicate
import logging
from time import time

# Set up logging
logging.basicConfig(level=logging.INFO)

# 환경 변수에서 토큰 가져오기
api_token = os.getenv("REPLICATE_API_TOKEN")
if not api_token:
    raise ValueError("API token is not set")

# Replicate 클라이언트 설정
client = replicate.Client(api_token=api_token)

# 이미지 처리 함수
def process_images(top_img_path, bottom_img_path, output_folder):
    # Check if the top image file exists
    if not os.path.exists(top_img_path):
        raise FileNotFoundError(f"The file {top_img_path} does not exist at the specified path.")

    # 상의 처리
    top_input = {
        "garm_img": open(top_img_path, "rb"),
        "human_img": open(r".\static\uploads\model_img.jpg", "rb"),
        "garment_des": "white shirt"
    }

    logging.info(f"Starting top processing for {output_folder}...")
    start_time = time()
    top_output = client.run(
        "cuuupid/idm-vton:c871bb9b046607b680449ecbae55fd8c6d945e0a1948644bf2361b3d021d3ff4",
        input=top_input
    )
    end_time = time()
    logging.info(f"Top processing completed in {end_time - start_time:.2f} seconds.")

    # 상의 결과 저장
    top_output_path = os.path.join(output_folder, "top_output.jpg")
    with open(top_output_path, "wb") as file:
        file.write(top_output.read())

    # Check if the bottom image file exists
    if not os.path.exists(bottom_img_path):
        raise FileNotFoundError(f"The file {bottom_img_path} does not exist at the specified path.")

    # 하의 처리
    bottom_input = {
        "category": "lower_body",
        "garm_img": open(bottom_img_path, "rb"),
        "human_img": open(top_output_path, "rb"),
        "garment_des": "white pants"
    }

    logging.info(f"Starting bottom processing for {output_folder}...")
    start_time = time()
    bottom_output = client.run(
        "cuuupid/idm-vton:c871bb9b046607b680449ecbae55fd8c6d945e0a1948644bf2361b3d021d3ff4",
        input=bottom_input
    )
    end_time = time()
    logging.info(f"Bottom processing completed in {end_time - start_time:.2f} seconds.")

    # 하의 결과 저장
    bottom_output_path = os.path.join(output_folder, "bottom_output.jpg")
    with open(bottom_output_path, "wb") as file:
        file.write(bottom_output.read())

def find_image_by_prefix(directory, prefix):
    """디렉토리에서 특정 접두사로 시작하는 첫 번째 이미지 파일을 찾습니다."""
    if not os.path.exists(directory):
        return None
    
    for filename in os.listdir(directory):
        if filename.upper().startswith(prefix.upper()) and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            return os.path.join(directory, filename)
    return None

# 이미지 경로 및 출력 폴더 설정
combinations = [
    (r".\static\images\result\combination\best_1", r".\static\images\result\combination\best_1", r".\static\images\result\combination\best_1"),
    (r".\static\images\result\combination\best_2", r".\static\images\result\combination\best_2", r".\static\images\result\combination\best_2"),
    (r".\static\images\result\combination\best_3", r".\static\images\result\combination\best_3", r".\static\images\result\combination\best_3")
]

# 각 조합에 대해 이미지 처리 수행
for top_dir, bottom_dir, output_folder in combinations:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # T로 시작하는 상의 이미지와 B로 시작하는 하의 이미지 찾기
    top_img = find_image_by_prefix(top_dir, 'T')
    bottom_img = find_image_by_prefix(bottom_dir, 'B')
    
    if not top_img:
        logging.error(f"No top image (starting with 'T') found in {top_dir}")
        continue
    
    if not bottom_img:
        logging.error(f"No bottom image (starting with 'B') found in {bottom_dir}")
        continue
        
    process_images(top_img, bottom_img, output_folder)
import cv2 
from PIL import Image
import numpy as np


def color_adjustment(source, target_mean=np.array([ 94.0, 137.0, 85.0]), image_size=512):
    """ Reference Image를 사용하여 color adjust 
    - General: np.array([ 100.0, 130.0, 90.0])
    - Future: np.array([ 94.0, 137.0, 85.0]) 
    """
    source = cv2.resize(source, (image_size, image_size))
    
    # 이미지를 LAB 색상 공간으로 변환
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)

    # 각 이미지의 L, A, B 채널 평균 계산
    source_mean = np.mean(source_lab, axis=(0, 1))

    # 채널별 차이 계산
    diff = target_mean - source_mean

    # 소스 이미지 조정
    adjusted_lab = source_lab + diff

    # 값 범위 제한 (0-255)
    adjusted_lab = np.clip(adjusted_lab, 0, 255).astype(np.uint8)

    # LAB에서 BGR로 다시 변환
    adjusted = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)

    return adjusted


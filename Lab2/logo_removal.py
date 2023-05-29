import matplotlib.pyplot as plt
import numpy as np
import cv2
from operator import itemgetter
import copy

from utils import draw_matches, get_inliers_homography, Ransac_DLT_homography, kp_match

def create_detector(name, nfeatures  = 2000):
    if name == 'orb':
        detector = cv2.ORB_create(nfeatures=nfeatures)
    else: detector = cv2.SIFT_create(nfeatures=nfeatures)   
    return detector

def get_features(img, detector_name = 'orb',):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = create_detector(detector_name)
    kps, descs = detector.detectAndCompute(gray, None)
    return kps, descs, img.shape[:2][::-1]

def features_to_kp(img, logo_features, ratio = .8, desc = 'orb', nmatches = 15):
    train_kps, train_descs, shape = logo_features
    kps, descs, _ = get_features(img, detector_name=desc)    
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(train_descs, descs, k=2)
    
    return kps, train_kps, matches, #sorted(matches, key = lambda x:x.distance)[:nmatches]


def match_template(template, target, margin = -20):
    
    res = cv2.matchTemplate(target,template, cv2.TM_CCOEFF)
    w, h, _ = template.shape

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w + margin, top_left[1] + h+margin)
    top_left = top_left[0] - margin, top_left[1] - margin
    target_blanked = np.zeros_like(target)
    
    cv2.rectangle(target_blanked,top_left, bottom_right, (255, 255, 255), -1)
    final_target = (0!=target_blanked) * target

    return target_blanked, final_target

def logo_substitution(template, target, new_logo):
    
    dsc = 'sift'
    template_ = copy.deepcopy(template)
    target_ = copy.deepcopy(target)
    new_logo = cv2.resize(new_logo, template.shape[:-1])
    mask, reduced_target = match_template(template_, target, margin = -5)
    kp1, kp2, matches_12 = kp_match(template, reduced_target, descr=dsc, dist_th=0.9) 
    P1, P2, H_12, idxs_inlier_matches_12 = get_inliers_homography(kp1, kp2, matches_12)
    #inlier_matches_12 = itemgetter(*idxs_inlier_matches_12)(matches_12)
    dst = cv2.warpPerspective(new_logo, H_12, target.shape[:-1][::-1])
    final = ~(dst * mask + target_ * ~mask)
    plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.savefig('out.png')

if __name__ == '__main__':

    template = cv2.imread('/home/adri/Desktop/master/M4/MCV-M4-3D-Vision/lab2/Data/logos/logoUPF.png', cv2.IMREAD_COLOR)
    target = cv2.imread('/home/adri/Desktop/master/M4/MCV-M4-3D-Vision/lab2/Data/logos/UPFbuilding.jpg', cv2.IMREAD_COLOR)
    subs = cv2.imread('/home/adri/Desktop/master/M4/MCV-M4-3D-Vision/lab2/Data/logos/logo_master.png', cv2.IMREAD_COLOR)

    logo_substitution(template, target, subs)

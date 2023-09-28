import torch
import numpy as np

class yolo_loss:

    def yolo_multitask_loss(y_pred, y_true): # 커스텀 손실함수. 배치 단위로 값이 들어온다
        
        batch_loss = 0
        count = len(y_true) # 모든 batch에 대해 값을 매겨야 한다. 
        for i in range(0, len(y_true)) :
            y_true_unit = y_true[i].clone().detach().requires_grad_(True)   # .requires_grad_(True)는 autograd에 모든 연산을 추적해야 한다고 알려준다. 
            y_pred_unit = y_pred[i].clone().detach().requires_grad_(True)   # clone().detach()를 하게 되면 새로운 텐서를 메모리에 할당하고 그것을 기존 계산 그래프와 끊어버린다. 

            y_true_unit = torch.reshape(y_true_unit, [49, 25])
            y_pred_unit = torch.reshape(y_pred_unit, [49, 30])
            
            loss = 0
            
            for j in range(0, len(y_true_unit)) : # 모든 grid에 대해 값을 보겠다. 
                # pred = [1, 30], true = [1, 25]
                
                bbox1_pred = y_pred_unit[j, :4].clone().detach().requires_grad_(True) 
                bbox1_pred_confidence = y_pred_unit[j, 4].clone().detach().requires_grad_(True)
                bbox2_pred = y_pred_unit[j, 5:9].clone().detach().requires_grad_(True)
                bbox2_pred_confidence = y_pred_unit[j, 9].clone().detach().requires_grad_(True)
                class_pred = y_pred_unit[j, 10:].clone().detach().requires_grad_(True)
                
                bbox_true = y_true_unit[j, :4].clone().detach().requires_grad_(True)
                bbox_true_confidence = y_true_unit[j, 4].clone().detach().requires_grad_(True)
                class_true = y_true_unit[j, 5:].clone().detach().requires_grad_(True)
                
                # IoU 구하기
                # x,y,w,h -> min_x, min_y, max_x, max_y로 변환
                box_pred_1_np = bbox1_pred.detach().numpy()
                box_pred_2_np = bbox2_pred.detach().numpy()
                box_true_np   = bbox_true.detach().numpy()

                box_pred_1_area = box_pred_1_np[2] * box_pred_1_np[3]
                box_pred_2_area = box_pred_2_np[2] * box_pred_2_np[3]
                box_true_area   = box_true_np[2]  * box_true_np[3]

                box_pred_1_minmax = np.asarray([box_pred_1_np[0] - 0.5*box_pred_1_np[2], box_pred_1_np[1] - 0.5*box_pred_1_np[3], box_pred_1_np[0] + 0.5*box_pred_1_np[2], box_pred_1_np[1] + 0.5*box_pred_1_np[3]])
                box_pred_2_minmax = np.asarray([box_pred_2_np[0] - 0.5*box_pred_2_np[2], box_pred_2_np[1] - 0.5*box_pred_2_np[3], box_pred_2_np[0] + 0.5*box_pred_2_np[2], box_pred_2_np[1] + 0.5*box_pred_2_np[3]])
                box_true_minmax   = np.asarray([box_true_np[0] - 0.5*box_true_np[2], box_true_np[1] - 0.5*box_true_np[3], box_true_np[0] + 0.5*box_true_np[2], box_true_np[1] + 0.5*box_true_np[3]])

                # 겹치는 영역의 (min_x, min_y, max_x, max_y)
                InterSection_pred_1_with_true = [max(box_pred_1_minmax[0], box_true_minmax[0]), max(box_pred_1_minmax[1], box_true_minmax[1]), min(box_pred_1_minmax[2], box_true_minmax[2]), min(box_pred_1_minmax[3], box_true_minmax[3])]
                InterSection_pred_2_with_true = [max(box_pred_2_minmax[0], box_true_minmax[0]), max(box_pred_2_minmax[1], box_true_minmax[1]), min(box_pred_2_minmax[2], box_true_minmax[2]), min(box_pred_2_minmax[3], box_true_minmax[3])]

                # 박스별로 IoU를 구한다
                IntersectionArea_pred_1_true = 0

                # 음수 * 음수 = 양수일 수도 있으니 검사를 한다.
                if (InterSection_pred_1_with_true[2] - InterSection_pred_1_with_true[0] + 1) >= 0 and (InterSection_pred_1_with_true[3] - InterSection_pred_1_with_true[1] + 1) >= 0 :
                        IntersectionArea_pred_1_true = (InterSection_pred_1_with_true[2] - InterSection_pred_1_with_true[0] + 1) * InterSection_pred_1_with_true[3] - InterSection_pred_1_with_true[1] + 1

                IntersectionArea_pred_2_true = 0

                if (InterSection_pred_2_with_true[2] - InterSection_pred_2_with_true[0] + 1) >= 0 and (InterSection_pred_2_with_true[3] - InterSection_pred_2_with_true[1] + 1) >= 0 :
                        IntersectionArea_pred_2_true = (InterSection_pred_2_with_true[2] - InterSection_pred_2_with_true[0] + 1) * InterSection_pred_2_with_true[3] - InterSection_pred_2_with_true[1] + 1

                Union_pred_1_true = box_pred_1_area + box_true_area - IntersectionArea_pred_1_true
                Union_pred_2_true = box_pred_2_area + box_true_area - IntersectionArea_pred_2_true

                IoU_box_1 = IntersectionArea_pred_1_true/Union_pred_1_true
                IoU_box_2 = IntersectionArea_pred_2_true/Union_pred_2_true
                            
                responsible_box = 0
                responsible_bbox_confidence = 0
                non_responsible_bbox_confidence = 0

                # box1, box2 중 responsible한걸 선택(IoU 기준)
                if IoU_box_1 >= IoU_box_2 :
                    responsible_box = bbox1_pred.clone().detach().requires_grad_(True)
                    responsible_bbox_confidence = bbox1_pred_confidence.clone().detach().requires_grad_(True)
                    non_responsible_bbox_confidence = bbox2_pred_confidence.clone().detach().requires_grad_(True)
                                    
                else :
                    responsible_box = bbox2_pred.clone().detach().requires_grad_(True)
                    responsible_bbox_confidence = bbox2_pred_confidence.clone().detach().requires_grad_(True)
                    non_responsible_bbox_confidence = bbox1_pred_confidence.clone().detach().requires_grad_(True)
                    
                # 1obj(i) 정하기(해당 셀에 객체의 중심좌표가 들어있는가?)
                obj_exist = torch.ones_like(bbox_true_confidence)
                if box_true_np[0] == 0.0 and box_true_np[1] == 0.0 and box_true_np[2] == 0.0 and box_true_np[3] == 0.0 : 
                    obj_exist = torch.zeros_like(bbox_true_confidence) 
                
                            
                # 만약 해당 cell에 객체가 없으면 confidence error의 no object 파트만 판단. (label된 값에서 알아서 해결)
                # 0~3 : bbox1의 위치 정보, 4 : bbox1의 bbox confidence score, 5~8 : bbox2의 위치 정보, 9 : bbox2의 confidence score, 10~29 : cell에 존재하는 클래스 확률 = pr(class | object) 

                # localization error 구하기(x,y,w,h). x, y는 해당 grid cell의 중심 좌표와 offset이고 w, h는 전체 이미지에 대해 정규화된 값이다. 즉, 범위가 0~1이다.
                localization_err_x = torch.pow( torch.subtract(bbox_true[0], responsible_box[0]), 2) # (x-x_hat)^2
                localization_err_y = torch.pow( torch.subtract(bbox_true[1], responsible_box[1]), 2) # (y-y_hat)^2

                localization_err_w = torch.pow( torch.subtract(torch.sqrt(bbox_true[2]), torch.sqrt(responsible_box[2])), 2) # (sqrt(w) - sqrt(w_hat))^2
                localization_err_h = torch.pow( torch.subtract(torch.sqrt(bbox_true[3]), torch.sqrt(responsible_box[3])), 2) # (sqrt(h) - sqrt(h_hat))^2
                
                # nan 방지
                if torch.isnan(localization_err_w).detach().numpy() == True :
                    localization_err_w = torch.zeros_like(localization_err_w)
                
                if torch.isnan(localization_err_h).detach().numpy() == True :
                    localization_err_h = torch.zeros_like(localization_err_h)
                
                localization_err_1 = torch.add(localization_err_x, localization_err_y)
                localization_err_2 = torch.add(localization_err_w, localization_err_h)
                localization_err = torch.add(localization_err_1, localization_err_2)
                
                weighted_localization_err = torch.multiply(localization_err, 5.0) # 5.0 : λ_coord
                weighted_localization_err = torch.multiply(weighted_localization_err, obj_exist) # 1obj(i) 곱하기
                
                # confidence error 구하기. true의 경우 답인 객체는 1 * ()고 아니면 0*()가 된다. 
                # index 4, 9에 있는 값(0~1)이 해당 박스에 객체가 있을 확률을 나타낸거다. Pr(obj in bbox)
                
                class_confidence_score_obj = torch.pow(torch.subtract(responsible_bbox_confidence, bbox_true_confidence), 2)
                class_confidence_score_noobj = torch.pow(torch.subtract(non_responsible_bbox_confidence, torch.zeros_like(bbox_true_confidence)), 2)
                class_confidence_score_noobj = torch.multiply(class_confidence_score_noobj, 0.5)
                
                class_confidence_score_obj = torch.mul(class_confidence_score_obj, obj_exist)
                class_confidence_score_noobj = torch.mul(class_confidence_score_noobj, torch.subtract(torch.ones_like(obj_exist), obj_exist)) # 객체가 존재하면 0, 존재하지 않으면 1을 곱합
                
                class_confidence_score = torch.add(class_confidence_score_obj,  class_confidence_score_noobj) 
                
                # classification loss(10~29. 인덱스 10~29에 해당되는 값은 Pr(Class_i|Object)이다. 객체가 cell안에 있을 때 해당 객체일 확률
                # class_true_oneCell는 진짜 객체의 인덱스에 해당하ㄴ 원소의 값만 1이고 나머지는 0 
                
                torch.pow(torch.subtract(class_true, class_pred), 2.0) # 여기서 에러
                
                classification_err = torch.pow(torch.subtract(class_true, class_pred), 2.0)
                classification_err = torch.sum(classification_err)
                classification_err = torch.multiply(classification_err, obj_exist)
                
                # loss합체
                loss_OneCell_1 = torch.add(weighted_localization_err, class_confidence_score)
                
                loss_OneCell = torch.add(loss_OneCell_1, classification_err)
                
                if loss == 0 :
                    loss = loss_OneCell.clone().detach().requires_grad_(True)
                else :
                    loss = torch.add(loss, loss_OneCell)
            
            if batch_loss == 0 :
                batch_loss = loss.clone().detach().requires_grad_(True)
            else :
                batch_loss = torch.add(batch_loss, loss)
            
        # 배치에 대한 loss 구하기
        batch_loss = torch.divide(batch_loss, count)
        
        return batch_loss
from base import db
from base.com.vo.detection_vo import DetectionVO
from base.com.vo.detection_images_vo import DetectionImagesVO


class DetectionDAO:
    def add_detection(self, detection_vo):
        db.session.add(detection_vo)
        db.session.commit()

    def view_detection(self):
        detection_vo_list=DetectionVO.query.all()
        return detection_vo_list

    def add_detection_images(self,detection_image_vo):
        db.session.add(detection_image_vo)
        db.session.commit()

    def view_detection_images(self):
        detection_images_vo_list=DetectionImagesVO.query.all()
        return detection_images_vo_list

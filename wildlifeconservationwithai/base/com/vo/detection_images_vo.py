from base import db

class DetectionImagesVO(db.Model):
    __tablename__ = 'detection_image_table'
    image_id = db.Column('image_id', db.Integer, primary_key=True,
                           autoincrement=True)

    image_name = db.Column('image_name', db.String(255),
                                   nullable=False)
    image_file_path = db.Column('image_file_path', db.String(255),
                                   nullable=False)
    detection_time= db.Column('detection_time', db.String(255),
                                   nullable=False)
    detection_date = db.Column('detection_date', db.String(255),
                               nullable=False)

db.create_all()

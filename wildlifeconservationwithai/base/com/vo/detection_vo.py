from base import db

class DetectionVO(db.Model):
    __tablename__ = 'detection_table'
    file_id = db.Column('file_id', db.Integer, primary_key=True,
                           autoincrement=True)

    file_name = db.Column('file_name', db.String(255),
                                   nullable=False)
    input_file_path = db.Column('input_file_path', db.String(255),
                                   nullable=False)
    output_file_path= db.Column('output_file_path', db.String(255),
                                   nullable=False)

db.create_all()

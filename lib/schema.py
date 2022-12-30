from marshmallow import Schema, fields, validate


class CheckValidation(Schema):
    upload_type = fields.Int(required=True, validate=validate.OneOf(choices=[1, 2]))
    image = fields.Str(required=True)

    # upload_type == 1이면 image_url (https:// 포함 여부 체크)
    # upload_type == 2이면 base64_image (base64: 어쩌고 포함여부 체크)
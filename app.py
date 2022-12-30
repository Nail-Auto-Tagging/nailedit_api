from http import HTTPStatus
from flask import Flask, request, abort
from flask_cors import CORS
from flask_restx import Api, Resource
from marshmallow import ValidationError
from lib.service import get_colors, get_categories
from lib.schema import CheckValidation


app = Flask(__name__)
api = Api(app)
CORS(app, resources={r"*": {"origins": "*"}})


@api.route("/color")
class Color(Resource):
    def get(self):
        req = request.args
        try:
            params = CheckValidation().load(req)
            result, status_code = get_colors(params)
        except ValidationError as e:
            abort(
                HTTPStatus.BAD_REQUEST,
                {"message": e.messages},
            )
        except Exception as e:
            abort(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"message": e.__str__()},
            )
        return result

    def post(self):
        req = request.get_json()
        try:
            body = CheckValidation().load(req)
            result, status_code = get_colors(body)
        except ValidationError as e:
            abort(
                HTTPStatus.BAD_REQUEST,
                {"message": e.messages},
            )
        except Exception as e:
            abort(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"message": e.__str__()},
            )
        return result


@api.route("/category")
class Category(Resource):
    def get(self):
        req = request.args
        try:
            params = CheckValidation().load(req)
            result, status_code = get_categories(params)
        except ValidationError as e:
            abort(
                HTTPStatus.BAD_REQUEST,
                {"message": e.messages},
            )
        except Exception as e:
            abort(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"message": e.__str__()},
            )
        return result

    def post(self):
        req = request.get_json()
        try:
            body = CheckValidation().load(req)
            result, status_code = get_categories(body)
        except ValidationError as e:
            abort(
                HTTPStatus.BAD_REQUEST,
                {"message": e.messages},
            )
        except Exception as e:
            abort(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"message": e.__str__()},
            )
        return result


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=80)

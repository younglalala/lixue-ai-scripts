from tornado.options import parse_command_line


parse_command_line()


DB = dict(
    # conn_param="mysql+pymysql://dev:lOKDtlfXegGNBi6O@rm-uf6v2zl126ld485x3o.mysql.rds.aliyuncs.com/lixue_test",
    conn_param="mysql+pymysql://dev:lOKDtlfXegGNBi6O@rm-uf6e9pd0u5f79gm85o.mysql.rds.aliyuncs.com/lixue?charset=utf8mb4",
    debug=False,
)

## API
To run the api.py written on fast API and that was able to pass all the intended tests but only manually run
`uvicorn api:app --reload`

### Notes regarding the stress and API tests.
it seems that the stress test has a bug related to a dependancy comming from flask, given that it fails when it tries to import escape from jinja2 despite the case that 'escape' is currently an unsupported package.

Also the 'anyio' package used on the API tests has a problem related to a broken attribute in this case is 'start_blocking_portal' which it seems to be related to the use of an outadet version of FastAPI.
Nevertheless all the tests written on the test_api.py passed when were runned manually on local.



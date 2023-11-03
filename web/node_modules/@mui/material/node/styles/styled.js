"use strict";
'use client';

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.slotShouldForwardProp = exports.rootShouldForwardProp = exports.default = void 0;
var _system = require("@mui/system");
var _defaultTheme = _interopRequireDefault(require("./defaultTheme"));
var _identifier = _interopRequireDefault(require("./identifier"));
const rootShouldForwardProp = prop => (0, _system.shouldForwardProp)(prop) && prop !== 'classes';
exports.rootShouldForwardProp = rootShouldForwardProp;
const slotShouldForwardProp = exports.slotShouldForwardProp = _system.shouldForwardProp;
const styled = (0, _system.createStyled)({
  themeId: _identifier.default,
  defaultTheme: _defaultTheme.default,
  rootShouldForwardProp
});
var _default = exports.default = styled;
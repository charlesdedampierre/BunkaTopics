# Installation
> `npm install --save @types/parse-json`

# Summary
This package contains type definitions for parse-json (https://github.com/sindresorhus/parse-json).

# Details
Files were exported from https://github.com/DefinitelyTyped/DefinitelyTyped/tree/master/types/parse-json.
## [index.d.ts](https://github.com/DefinitelyTyped/DefinitelyTyped/tree/master/types/parse-json/index.d.ts)
````ts
declare function parseJson(input: string | null, filepath?: string): any;
declare function parseJson(input: string | null, reviver: (key: any, value: any) => any, filepath?: string): any;

export = parseJson;

````

### Additional Details
 * Last updated: Wed, 18 Oct 2023 05:47:08 GMT
 * Dependencies: none

# Credits
These definitions were written by [mrmlnc](https://github.com/mrmlnc).

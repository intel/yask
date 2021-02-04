/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2021, Intel Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

* The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

*****************************************************************************/

///////// String-parsing support for cmd-line options.

#pragma once

#include <cmath>
#include <iostream>
#include <iomanip>
#include <memory>
#include <algorithm>
#include <sstream>
#include <functional>
#include <map>
#include <set>
#include <vector>

using namespace std;

namespace yask {

    class ArgParser {
    public:
    	virtual ~ArgParser() {}

        // For strings like "x,y", call the lambda function for
        // each element.
        virtual void parse_list(const string& arg_str,
                               function<void (const string& key)> handle)
        {
            if (arg_str.length() == 0)
                return;

            // split by commas.
            vector<string> args;
            string arg;
            for (char c1 : arg_str) {
                if (c1 == ',') {
                    args.push_back(arg);
                    arg = "";
                } else
                    arg += c1;
            }
            args.push_back(arg);

            // process each element.
            for (auto p_str : args) {

                // call handler.
                handle(p_str);
            }
        }

        // For strings like "x=4,y=2", call the lambda function for
        // each key, value pair.
        virtual void parse_key_value_pairs(const string& arg_str,
                                        function<void (const string& key,
                                                       const string& value)> handle)
        {
            parse_list
                (arg_str,
                 [&](const string& p_str) { 
            
                     // split by equal sign.
                     size_t ep = p_str.find("=");
                     if (ep == string::npos) {
                         THROW_YASK_EXCEPTION("Error: no equal sign in '" + p_str + "'");
                     }
                     string key = p_str.substr(0, ep);
                     string value = p_str.substr(ep+1);

                     // call handler.
                     handle(key, value);
                 } );
        }
    };

} // namespace yask.


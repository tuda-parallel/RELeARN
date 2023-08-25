/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_string_util.h"

#include "adapter/random/RandomAdapter.h"

#include "util/StringUtil.h"

TEST_F(StringUtilTest, testFormatException) {
    auto random_int = RandomAdapter::get_random_integer<int>(1, 999, mt);
    ASSERT_THROW(auto val = StringUtil::format_int_with_leading_zeros(random_int, 0);, RelearnException);
}

TEST_F(StringUtilTest, testFormatFill) {
    auto check = [](auto num, auto digits, auto descr) {
        const auto formatted = StringUtil::format_int_with_leading_zeros(num, digits);
        ASSERT_EQ(descr, formatted);
    };

    check(0, 2, "00");
    check(1, 2, "01");
    check(2, 2, "02");
    check(3, 2, "03");
    check(4, 2, "04");
    check(5, 2, "05");
    check(6, 2, "06");
    check(7, 2, "07");
    check(8, 2, "08");
    check(9, 2, "09");

    check(0, 3, "000");
    check(1, 3, "001");
    check(2, 3, "002");
    check(3, 3, "003");
    check(4, 3, "004");
    check(5, 3, "005");
    check(6, 3, "006");
    check(7, 3, "007");
    check(8, 3, "008");
    check(9, 3, "009");

    check(0, 7, "0000000");
    check(1, 7, "0000001");
    check(2, 7, "0000002");
    check(3, 7, "0000003");
    check(4, 7, "0000004");
    check(5, 7, "0000005");
    check(6, 7, "0000006");
    check(7, 7, "0000007");
    check(8, 7, "0000008");
    check(9, 7, "0000009");

    check(5410642, 8, "05410642");
    check(5411642, 8, "05411642");
    check(5412642, 8, "05412642");
    check(5413642, 8, "05413642");
    check(5414642, 8, "05414642");
    check(5415642, 8, "05415642");
    check(5416642, 8, "05416642");
    check(5417642, 8, "05417642");
    check(5418642, 8, "05418642");
    check(5419642, 8, "05419642");
}

TEST_F(StringUtilTest, testFormatNoFill) {
    auto check = [](auto num, auto digits, auto descr) {
        const auto formatted = StringUtil::format_int_with_leading_zeros(num, digits);
        ASSERT_EQ(descr, formatted);
    };

    check(0, 1, "0");
    check(1, 1, "1");
    check(2, 1, "2");
    check(3, 1, "3");
    check(4, 1, "4");
    check(5, 1, "5");
    check(6, 1, "6");
    check(7, 1, "7");
    check(8, 1, "8");
    check(9, 1, "9");

    check(32, 2, "32");
    check(39, 2, "39");
    check(65, 2, "65");
    check(53, 2, "53");
    check(13, 2, "13");
    check(72, 2, "72");
    check(95, 2, "95");
    check(89, 2, "89");
    check(32, 2, "32");
    check(23, 2, "23");

    check(303, 3, "303");
    check(709, 3, "709");
    check(628, 3, "628");
    check(832, 3, "832");
    check(221, 3, "221");
    check(705, 3, "705");
    check(751, 3, "751");
    check(478, 3, "478");
    check(359, 3, "359");
    check(877, 3, "877");

    check(5927, 4, "5927");
    check(7476, 4, "7476");
    check(4041, 4, "4041");
    check(1774, 4, "1774");
    check(2593, 4, "2593");
    check(5438, 4, "5438");
    check(4319, 4, "4319");
    check(6947, 4, "6947");
    check(8510, 4, "8510");
    check(9998, 4, "9998");

    check(18019, 5, "18019");
    check(58587, 5, "58587");
    check(44734, 5, "44734");
    check(36505, 5, "36505");
    check(44333, 5, "44333");
    check(31948, 5, "31948");
    check(36589, 5, "36589");
    check(19340, 5, "19340");
    check(79639, 5, "79639");
    check(35187, 5, "35187");
}

TEST_F(StringUtilTest, testFormatTooLong) {
    auto check = [](auto num, auto digits, auto descr) {
        const auto formatted = StringUtil::format_int_with_leading_zeros(num, digits);
        ASSERT_EQ(descr, formatted);
    };

    check(32, 1, "32");
    check(39, 1, "39");
    check(65, 1, "65");
    check(53, 1, "53");
    check(13, 1, "13");
    check(72, 1, "72");
    check(95, 1, "95");
    check(89, 1, "89");
    check(32, 1, "32");
    check(23, 1, "23");

    check(303, 2, "303");
    check(709, 2, "709");
    check(628, 2, "628");
    check(832, 2, "832");
    check(221, 2, "221");
    check(705, 2, "705");
    check(751, 2, "751");
    check(478, 2, "478");
    check(359, 2, "359");
    check(877, 2, "877");

    check(5927, 3, "5927");
    check(7476, 3, "7476");
    check(4041, 3, "4041");
    check(1774, 3, "1774");
    check(2593, 3, "2593");
    check(5438, 3, "5438");
    check(4319, 3, "4319");
    check(6947, 3, "6947");
    check(8510, 3, "8510");
    check(9998, 3, "9998");

    check(18019, 4, "18019");
    check(58587, 4, "58587");
    check(44734, 4, "44734");
    check(36505, 4, "36505");
    check(44333, 4, "44333");
    check(31948, 4, "31948");
    check(36589, 4, "36589");
    check(19340, 4, "19340");
    check(79639, 4, "79639");
    check(35187, 4, "35187");
}

TEST_F(StringUtilTest, testStringSplitEmpty) {
    const std::string original = "";
    const auto& split = StringUtil::split_string(original, 'k');

    ASSERT_TRUE(split.empty());
}

TEST_F(StringUtilTest, testStringSplitNoDelim) {
    const std::string original = "gukztqns2moa,y.l#sdp+cawö34i3zc\"!$%)=&&%\"\"!°";

    const auto& split = StringUtil::split_string(original, 'x');

    ASSERT_EQ(split.size(), 1);
    ASSERT_EQ(split[0], "gukztqns2moa,y.l#sdp+cawö34i3zc\"!$%)=&&%\"\"!°");
}

TEST_F(StringUtilTest, testStringSplitOneDelim) {
    const std::string original = "0";

    const auto& split = StringUtil::split_string(original, '0');

    ASSERT_EQ(split.size(), 1);
    ASSERT_EQ(split[0], "");
}

TEST_F(StringUtilTest, testStringSplit) {
    const std::string original = ";hfke;kg83;;058372;058372;4sfsaf ;iousahfu-+30o3q;021u3zhrns;";

    const auto& split = StringUtil::split_string(original, ';');

    ASSERT_EQ(split.size(), 9);
    ASSERT_EQ(split[0], "");
    ASSERT_EQ(split[1], "hfke");
    ASSERT_EQ(split[2], "kg83");
    ASSERT_EQ(split[3], "");
    ASSERT_EQ(split[4], "058372");
    ASSERT_EQ(split[5], "058372");
    ASSERT_EQ(split[6], "4sfsaf ");
    ASSERT_EQ(split[7], "iousahfu-+30o3q");
    ASSERT_EQ(split[8], "021u3zhrns");
}
